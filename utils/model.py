import torch
import torch.nn as nn

class BioClipClassifier(nn.Module):
    def __init__(self, vision_model, num_classes, noise_alpha=0.2, dropout_rate=0.2):
        super(BioClipClassifier, self).__init__()
        self.vision_model = vision_model
        self.embedding_dim = vision_model.output_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.embedding_dim, num_classes)
        self.noise_alpha = noise_alpha

    def forward(self, x):
        image_embeddings = self.vision_model(x)
        if self.training:
            dims = torch.tensor(self.embedding_dim, dtype=torch.float)
            mag_norm = self.noise_alpha / torch.sqrt(dims)
            noise = torch.zeros_like(image_embeddings).uniform_(-mag_norm, mag_norm)
            image_embeddings = image_embeddings + noise
        image_embeddings = self.dropout(image_embeddings)
        logits = self.classifier(image_embeddings)
        return logits