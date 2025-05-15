# BioCLIP Aquatic Invertebrate Classifier
Co-author: [Abdelaziz Ibrahim](https://github.com/AbdulazizMahmood)

Fine-tuning [BioCLIP](https://huggingface.co/imageomics/bioclip) for aquatic invertebrate image classification!  
Dataset: [Aquamonitor-JYU](https://huggingface.co/datasets/mikkoim/aquamonitor-jyu)

**Model link coming soon!**

## 🚀 Training Details

- **Model:** BioCLIP (ViT-B/16 backbone)
- **Dataset:** [Aquamonitor-JYU](https://huggingface.co/datasets/mikkoim/aquamonitor-jyu)
- **Task:** Multi-class image classification

### Two-Stage Training

1. **Stage 1:**  
   - Freeze vision encoder  
   - Train classifier head only  
   - 4 epochs, LR = `1e-3`, weight decay = `0.01`

2. **Stage 2:**  
   - Unfreeze last 5 transformer blocks  
   - Train classifier head + last 5 blocks  
   - 4 epochs, Cosine LR decay: `1e-3` → `4e-4`, weight decay = `0.05`

### Parameter-Efficient

- **LoRA:**  
  - Rank `r=64`, alpha `64`  
  - Applied to attention output & MLP layers (last 5 blocks)
- **DoRA:**  
  - Decompose weights (magnitude & direction)

### Data Augmentation

- Random resized crop (`scale=0.7-1.0`)
- Horizontal flip (`p=0.5`)
- Color jitter (brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
- Gaussian blur (`p=0.2`)
- Sharpness (`p=0.1`)
- Random erasing (`p=0.1`)

### Other Training Details

- **NEFTune-like:** Add noise to image embeddings (alpha = `1.0`)
- **Loss:** Cross-entropy with inverse frequency class weights
- **Optimizer:** AdamW
- **Dropout:** 0.2 (classifier)
- **Gradient clipping:** 0.5
- **Batch size:** 64

---

## 📊 BioCLIP Results

| Split        | Accuracy | F1 (Weighted) | Precision (Weighted) | Recall (Weighted) |
|--------------|----------|---------------|----------------------|-------------------|
| **Validation** | **0.8025** | **0.7943**      | **0.8174**             | **0.8025**          |
| **Training**   | 0.9360   | 0.9361        | 0.9367               | 0.9360            |
