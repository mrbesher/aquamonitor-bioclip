import datetime
import json
import os
import shutil
from collections import defaultdict

import open_clip
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from utils.data import (
    collate_fn,
    download_class_descriptions,
    get_transforms,
    load_class_descriptions,
    load_metadata,
    prepare_class_maps,
    preprocess_train,
    preprocess_val,
)
from utils.logger import TrainingLogger
from utils.metrics import compute_metrics
from utils.model import BioClipClassifier
from utils.train import evaluate, run_training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
NUM_EPOCHS_STAGE1 = 4
NUM_EPOCHS_STAGE2 = 6
LEARNING_RATE = 1e-3
MIN_LR = 4e-4
LR_REDUCTION_FACTOR = 0.8
N_UNFREEZE_LAYERS = 5
LORA_ALPHA = 64
LORA_RANK = 64
LORA_DROPOUT = 0.4
NOISE_ALPHA = 1.0
GRAD_CLIP = 0.5

download_class_descriptions()
class_descriptions = load_class_descriptions()
ds, metadata = load_metadata()
classes, class_map, class_map_inv, label_dict = prepare_class_maps(metadata)

print("Loading BioClip model...")
model_base, preprocess_train_base, preprocess_val_base = (
    open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
)
tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
train_transforms, preprocess_val_base = get_transforms(
    preprocess_train_base, preprocess_val_base
)


def preprocess_train_fn(batch):
    return preprocess_train(batch, train_transforms, label_dict)


def preprocess_val_fn(batch):
    return preprocess_val(batch, preprocess_val_base, label_dict)


print("Preparing datasets...")
train_ds = ds["train"].with_transform(preprocess_train_fn)
val_ds = ds["validation"].with_transform(preprocess_val_fn)
train_dataloader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True
)
val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

value_counts = metadata["taxon_group"].value_counts()
n_samples = value_counts.sum()
weights = {k: n_samples / (len(classes) * v) for k, v in value_counts.items()}
class_weights = {
    class_map[class_name]: weight for class_name, weight in weights.items()
}
class_weights_tensor = (
    torch.tensor([class_weights[idx] for idx in range(len(classes))]).float().to(device)
)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

wandb_api_key = os.environ.get("WANDB_API_KEY", None)
wandb_entity = os.environ.get("WANDB_ENTITY", None)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"bioclip_b{BATCH_SIZE}_lr{LEARNING_RATE}_e{NUM_EPOCHS_STAGE1}+{NUM_EPOCHS_STAGE2}_lora{LORA_RANK}r{N_UNFREEZE_LAYERS}l_na{NOISE_ALPHA}"
logger = TrainingLogger(
    log_dir="./logs/bioclip_classifier",
    project_name="aquamonitor_classification",
    experiment_name=experiment_name,
    backends=["tensorboard", "wandb"],
    wandb_api_key=wandb_api_key,
    wandb_entity=wandb_entity,
)

experiment_config = {
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "min_learning_rate": MIN_LR,
    "lr_reduction_factor": LR_REDUCTION_FACTOR,
    "num_epochs_stage1": NUM_EPOCHS_STAGE1,
    "num_epochs_stage2": NUM_EPOCHS_STAGE2,
    "grad_clip": GRAD_CLIP,
    "model_name": "imageomics/bioclip",
    "model_type": "BioClip Classifier",
    "embedding_dim": model_base.visual.output_dim,
    "num_classes": len(classes),
    "dataset": "aquamonitor-jyu",
    "train_samples": len(train_ds),
    "val_samples": len(val_ds),
    "class_count": len(classes),
    "class_distribution_min": int(value_counts.min()),
    "class_distribution_max": int(value_counts.max()),
    "class_distribution_mean": float(value_counts.mean()),
    "class_distribution_median": float(value_counts.median()),
    "aug_random_crop_scale_min": 0.7,
    "aug_random_crop_scale_max": 1.0,
    "aug_color_jitter_brightness": 0.3,
    "aug_color_jitter_contrast": 0.3,
    "aug_color_jitter_saturation": 0.2,
    "aug_color_jitter_hue": 0.1,
    "aug_gaussian_blur_prob": 0.2,
    "aug_sharpness_prob": 0.1,
    "aug_random_erasing_prob": 0.1,
    "aug_horizontal_flip_prob": 0.5,
    "dropout_rate": 0.2,
    "noise_alpha": NOISE_ALPHA,
    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "lora_use_dora": True,
    "unfreeze_layers": N_UNFREEZE_LAYERS,
    "device": str(device),
    "torch_version": torch.__version__,
    "timestamp": timestamp,
    "experiment_name": experiment_name,
}

logger.log_hyperparams(experiment_config)

model = BioClipClassifier(model_base.visual, len(classes), noise_alpha=NOISE_ALPHA)
model = model.to(device)
logger.log_model(model)

print("=" * 50)
print("Stage 1: Training classifier only")
print("=" * 50)
for param in model.vision_model.parameters():
    param.requires_grad = False
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE)
history_stage1 = run_training(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    criterion=loss_fn,
    device=device,
    num_epochs=NUM_EPOCHS_STAGE1,
    grad_clip=GRAD_CLIP,
    metric_fn=lambda: compute_metrics(num_classes=len(classes), device=device),
    model_path="bioclip_classifier_stage1.pt",
    monitor_metric="accuracy",
    logger=logger,
)

print("=" * 50)
print("Stage 2: Fine-tuning late vision layers and classifier")
print("=" * 50)
num_layers = 12
last_layer_numbers = [num_layers - i - 1 for i in range(N_UNFREEZE_LAYERS)]
layer_pattern = "|".join(str(i) for i in last_layer_numbers)
target_modules = rf"transformer\.resblocks\.({layer_pattern})\.(attn\.out_proj|mlp\.c_fc|mlp\.c_proj)"
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    use_dora=True,
    target_modules=target_modules,
)
model.vision_model = get_peft_model(model.vision_model, lora_config)
model.vision_model.print_trainable_parameters()
lora_config_dict = {
    k: v
    for k, v in lora_config.to_dict().items()
    if isinstance(v, (int, float, str, bool))
}
logger.log_hyperparams(lora_config_dict)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE * LR_REDUCTION_FACTOR,
    weight_decay=0.05,
)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_STAGE2, eta_min=MIN_LR)
history_stage2 = run_training(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    criterion=loss_fn,
    device=device,
    num_epochs=NUM_EPOCHS_STAGE2,
    grad_clip=GRAD_CLIP,
    metric_fn=lambda: compute_metrics(num_classes=len(classes), device=device),
    scheduler=scheduler,
    model_path="bioclip_classifier_finetuned.pt",
    monitor_metric="accuracy",
    logger=logger,
)
model.vision_model = model.vision_model.merge_and_unload()
logger.close()

print("=" * 50)
print("Final evaluation")
print("=" * 50)
final_val_loss, final_val_metrics = evaluate(
    model,
    val_dataloader,
    loss_fn,
    device,
    lambda: compute_metrics(num_classes=len(classes), device=device),
)
print(f"Final validation accuracy: {final_val_metrics['accuracy']:.4f}")
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_map": class_map,
        "class_map_inv": class_map_inv,
    },
    "bioclip_classifier_finetuned.pt",
)
print("Model saved to 'bioclip_classifier_finetuned.pt'")


def compute_per_class_metrics(model, dataloader, device, classes, class_map_inv):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    y_true = [class_map_inv[label] for label in all_labels]
    y_pred = [class_map_inv[pred] for pred in all_preds]
    class_accuracy = defaultdict(float)
    for true_class in classes:
        indices = [i for i, label in enumerate(y_true) if label == true_class]
        if indices:
            correct = sum(1 for i in indices if y_pred[i] == y_true[i])
            class_accuracy[true_class] = correct / len(indices)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )
    metrics = {
        "per_class_accuracy": dict(class_accuracy),
        "overall_accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics


print("Computing detailed metrics...")
detailed_metrics = compute_per_class_metrics(
    model, val_dataloader, device, classes, class_map_inv
)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_dir = f"./models/{timestamp}_BioClipClassifier"
os.makedirs(model_save_dir, exist_ok=True)
metadata_json = {
    "timestamp": timestamp,
    "model_type": "BioClipClassifier",
    "base_model": "imageomics/bioclip",
    "training_params": {
        "batch_size": BATCH_SIZE,
        "epochs_stage1": NUM_EPOCHS_STAGE1,
        "epochs_stage2": NUM_EPOCHS_STAGE2,
        "learning_rate": LEARNING_RATE,
        "device": str(device),
    },
    "results": {
        "final_validation_accuracy": final_val_metrics["accuracy"],
        "final_validation_loss": final_val_loss,
        "final_validation_precision": final_val_metrics["precision"],
        "final_validation_recall": final_val_metrics["recall"],
        "final_validation_f1": final_val_metrics["f1"],
        "per_class_accuracy": detailed_metrics["per_class_accuracy"],
        "overall_metrics": {
            "accuracy": detailed_metrics["overall_accuracy"],
            "precision": float(detailed_metrics["precision"]),
            "recall": float(detailed_metrics["recall"]),
            "f1": float(detailed_metrics["f1"]),
        },
    },
    "class_mapping": class_map,
}
model_path = os.path.join(model_save_dir, "bioclip_classifier_model.pt")
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_map": class_map,
        "class_map_inv": class_map_inv,
        "timestamp": timestamp,
    },
    model_path,
)
metadata_path = os.path.join(model_save_dir, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata_json, f, indent=4)
if os.path.exists("bioclip_classifier_stage1.pt"):
    shutil.copy(
        "bioclip_classifier_stage1.pt",
        os.path.join(model_save_dir, "bioclip_classifier_stage1.pt"),
    )
if os.path.exists("bioclip_classifier_finetuned.pt"):
    shutil.copy(
        "bioclip_classifier_finetuned.pt",
        os.path.join(model_save_dir, "bioclip_classifier_finetuned.pt"),
    )
print(f"Model and metadata saved to {model_save_dir}")
