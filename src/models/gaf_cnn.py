import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

def compute_gadf(x_tensor):
    """
    Computes Gramian Angular Difference Field (GADF) for multi-channel sensor time-series.
    Input: x_tensor of shape (batch, sensors, time_len)
    Output: GADF of shape (batch, sensors, time_len, time_len)
    """
    # Min-max scale per sensor window to [-1, 1]
    x_min = x_tensor.min(dim=-1, keepdim=True)[0]
    x_max = x_tensor.max(dim=-1, keepdim=True)[0]
    diff = x_max - x_min
    diff[diff == 0] = 1e-6
    x_norm = 2.0 * (x_tensor - x_min) / diff - 1.0
    x_norm = torch.clamp(x_norm, -1.0, 1.0)
    
    phi = torch.acos(x_norm) # (B, S, T)
    sin_phi = torch.sin(phi)
    cos_phi = x_norm
    
    # GADF = sin(phi_i - phi_j) = sin(phi_i)*cos(phi_j) - cos(phi_i)*sin(phi_j)
    gadf = torch.matmul(sin_phi.unsqueeze(-1), cos_phi.unsqueeze(-2)) - torch.matmul(cos_phi.unsqueeze(-1), sin_phi.unsqueeze(-2))
    return gadf

class GAF_2DCNN(nn.Module):
    """
    2D-CNN architecture for Gramian Angular Field (GAF) E-Nose classification.
    Processes 6-channel 2D spatial-temporal image maps (6, 64, 64).
    """
    def __init__(self, in_channels=6, num_classes=4):
        super(GAF_2DCNN, self).__init__()
        
        # Block 1: 6 -> 32 (64x64 -> 32x32)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(2)
        
        # Block 2: 32 -> 64 (32x32 -> 16x16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool2d(2)
        
        # Block 3: 64 -> 128 (16x16 -> 8x8)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.LeakyReLU(0.1)
        self.pool3 = nn.MaxPool2d(2)
        
        # Global Pooling + Dense
        self.gap = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes) # Raw logits for BCEWithLogitsLoss
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        logits = self.fc2(x)
        return logits

def train_gaf_cnn(X_raw, y_labels, epochs=35, batch_size=32, lr=0.001):
    """
    Trains the GAF 2D-CNN model.
    X_raw: (N, 6, 64)
    y_labels: (N, 4)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_raw, y_labels, test_size=0.2, random_state=42)
    
    # Compute GADF
    print("Converting sensor time-series to Gramian Angular Difference Fields (GADF)...")
    with torch.no_grad():
        X_tr_gadf = compute_gadf(torch.tensor(X_tr, dtype=torch.float32))
        X_val_gadf = compute_gadf(torch.tensor(X_val, dtype=torch.float32))
    
    print(f"GADF Train shape: {X_tr_gadf.shape}, Val shape: {X_val_gadf.shape}")
    
    train_dataset = TensorDataset(X_tr_gadf, torch.tensor(y_tr, dtype=torch.float32))
    val_dataset = TensorDataset(X_val_gadf, torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = GAF_2DCNN(in_channels=6, num_classes=4).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_loss = float('inf')
    best_weights = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                val_loss += criterion(logits, by).item()
                
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.state_dict().copy()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} - Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
            
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_gadf.to(device))
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        
    return model, val_probs, y_val
