import os
import json
import gdown
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import torch
from torchvision.transforms import v2 as T

def download_class_descriptions():
    if not os.path.exists('class_descriptions.json'):
        url = 'https://drive.google.com/file/d/18yKLlqCem64mE5jqc0WHreqiaj5rDWUy/view?usp=sharing'
        gdown.download(url, 'class_descriptions.json', fuzzy=True)

def load_class_descriptions():
    class_descriptions = {}
    with open('class_descriptions.json', 'r') as f:
        for line in f:
            item = json.loads(line)
            class_descriptions[item['name']] = item['description']
    return class_descriptions

def load_metadata():
    ds = load_dataset("mikkoim/aquamonitor-jyu")
    if not os.path.exists('aquamonitor-jyu.parquet.gzip'):
        hf_hub_download(
            repo_id="mikkoim/aquamonitor-jyu",
            filename="aquamonitor-jyu.parquet.gzip",
            repo_type="dataset",
            local_dir="."
        )
    metadata = pd.read_parquet('aquamonitor-jyu.parquet.gzip')
    return ds, metadata

def prepare_class_maps(metadata):
    classes = sorted(metadata["taxon_group"].unique())
    class_map = {k: v for v, k in enumerate(classes)}
    class_map_inv = {v: k for k, v in class_map.items()}
    metadata["img"] = metadata["img"].str.removesuffix(".jpg")
    label_dict = dict(zip(metadata["img"], metadata["taxon_group"].map(class_map)))
    return classes, class_map, class_map_inv, label_dict

def get_transforms(preprocess_train_base, preprocess_val_base):
    train_transforms = T.Compose([
        T.RandomResizedCrop(size=224, scale=(0.7, 1.0), ratio=(0.75, 1.3333), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
        T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
        preprocess_train_base,
        T.RandomErasing(p=0.1, scale=(0.02, 0.1))
    ])
    return train_transforms, preprocess_val_base

def preprocess_train(batch, train_transforms, label_dict):
    augmented_images = [train_transforms(img) for img in batch["jpg"]]
    return {
        "key": batch["__key__"],
        "img": torch.stack(augmented_images),
        "label": torch.as_tensor([label_dict[x] for x in batch["__key__"]], dtype=torch.long)
    }

def preprocess_val(batch, preprocess_val_base, label_dict):
    processed_images = [preprocess_val_base(img) for img in batch["jpg"]]
    return {
        "key": batch["__key__"],
        "img": torch.stack(processed_images),
        "label": torch.as_tensor([label_dict[x] for x in batch["__key__"]], dtype=torch.long)
    }

def collate_fn(batch):
    images = torch.stack([item['img'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return images, labels