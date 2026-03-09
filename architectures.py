import torch
import torch.nn as nn
import numpy as np

class ViTFeatureExtractor(nn.Module):
    """Mock Vision Transformer for feature extraction"""
    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        
    def forward(self, x):
        # x is assumed to be B x C x H x W
        batch_size = x.size(0)
        # Returns a mock embedding
        return torch.rand(batch_size, self.embed_dim)

class UNetLesionDetector(nn.Module):
    """Mock U-Net for lesion segmentation"""
    def __init__(self, in_channels=3, out_classes=1):
        super().__init__()
        
    def forward(self, x):
        batch_size, _, h, w = x.shape
        # Returns a mock probability mask for lesions
        return torch.rand(batch_size, 1, h, w)

class LSTMPredictor(nn.Module):
    """Mock LSTM for disease progression over time"""
    def __init__(self, input_dim=768, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1) # Outputs a risk increment score
        
    def forward(self, x):
        # x shape: B x TimeSteps x Features
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return torch.sigmoid(self.fc(last_out))
