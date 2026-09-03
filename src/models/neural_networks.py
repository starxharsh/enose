import torch
import torch.nn as torch_nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np

class ANN(torch_nn.Module):
    def __init__(self, input_dim: int, output_dim: int, task: str = 'classification'):
        super().__init__()
        self.task = task
        self.net = torch_nn.Sequential(
            torch_nn.Linear(input_dim, 256),
            torch_nn.BatchNorm1d(256),
            torch_nn.ReLU(),
            torch_nn.Dropout(0.3),
            torch_nn.Linear(256, 128),
            torch_nn.BatchNorm1d(128),
            torch_nn.ReLU(),
            torch_nn.Dropout(0.3),
            torch_nn.Linear(128, 64),
            torch_nn.BatchNorm1d(64),
            torch_nn.ReLU(),
            torch_nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.net(x)
        return out

class CNN1D(torch_nn.Module):
    def __init__(self, in_channels=6, output_dim=4, task='classification'):
        super().__init__()
        self.task = task
        self.in_channels = in_channels
        self.conv1 = torch_nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = torch_nn.BatchNorm1d(32)
        self.conv2 = torch_nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch_nn.BatchNorm1d(64)
        self.conv3 = torch_nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = torch_nn.BatchNorm1d(128)
        self.pool = torch_nn.MaxPool1d(2)
        self.adaptive_pool = torch_nn.AdaptiveAvgPool1d(1)
        self.fc1 = torch_nn.Linear(128, 64)
        self.fc2 = torch_nn.Linear(64, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), self.in_channels, -1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

class Attention(torch_nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = torch_nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        attn_weights = F.softmax(self.attention(lstm_output).squeeze(-1), dim=-1)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context_vector

class LSTM_Attention(torch_nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=4, task='classification'):
        super().__init__()
        self.task = task
        self.lstm = torch_nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)
        self.fc1 = torch_nn.Linear(hidden_dim * 2, 64)
        self.fc2 = torch_nn.Linear(64, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            # Reshape (batch, 126) -> (batch, 21, 6)
            x = x.view(x.size(0), -1, 6)
        elif x.dim() == 3 and x.size(1) == 6:
            x = x.transpose(1, 2)
            
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        x = F.relu(self.fc1(attn_out))
        out = self.fc2(x)
        return out

class ResidualBlock(torch_nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = torch_nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.prelu = torch_nn.PReLU()
        self.res_conv = torch_nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else torch_nn.Identity()

    def forward(self, x):
        out = self.prelu(self.conv(x))
        return out + self.res_conv(x)

class TCN_MHA(torch_nn.Module):
    def __init__(self, in_channels=6, output_dim=4, task='classification'):
        super().__init__()
        self.task = task
        self.in_channels = in_channels
        dilations = [1, 2, 4, 8]
        layers = []
        channels = in_channels
        for d in dilations:
            layers.append(ResidualBlock(channels, 32, d))
            channels = 32
        self.tcn = torch_nn.Sequential(*layers)
        
        self.mha = torch_nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.adaptive_pool = torch_nn.AdaptiveAvgPool1d(1)
        self.fc = torch_nn.Linear(32, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), self.in_channels, -1)
        x = self.tcn(x)
        x_mha = x.transpose(1, 2)
        attn_out, _ = self.mha(x_mha, x_mha, x_mha)
        attn_out = attn_out.transpose(1, 2)
        x = self.adaptive_pool(attn_out).squeeze(-1)
        out = self.fc(x)
        return out

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=10, task='classification'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = torch_nn.BCELoss() if task == 'classification' else torch_nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        correct_train, total_train = 0, 0
        
        for X, y in train_loader:
            X, y = X.to(device).float(), y.to(device).float()
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            if task == 'classification':
                preds = (out > 0.5).float()
                correct_train += (preds == y).sum().item()
                total_train += y.numel()
        
        train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(train_loss)
        if task == 'classification':
            train_metrics.append(correct_train / total_train)
        
        model.eval()
        epoch_val_loss = 0
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device).float(), y.to(device).float()
                out = model(X)
                loss = criterion(out, y)
                epoch_val_loss += loss.item()
                if task == 'classification':
                    preds = (out > 0.5).float()
                    correct_val += (preds == y).sum().item()
                    total_val += y.numel()
        
        val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if task == 'classification':
            val_metrics.append(correct_val / total_val)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_metrics[-1]:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_metrics[-1]:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), f"best_{model.__class__.__name__}.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered")
                break
                
    model.load_state_dict(torch.load(f"best_{model.__class__.__name__}.pth"))
    
    os.makedirs("results", exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title(f'{model.__class__.__name__} Loss Curve')
    plt.savefig(f'results/{model.__class__.__name__}_loss.png')
    
    if task == 'classification':
        plt.figure()
        plt.plot(train_metrics, label='Train Acc')
        plt.plot(val_metrics, label='Val Acc')
        plt.legend()
        plt.title(f'{model.__class__.__name__} Accuracy Curve')
        plt.savefig(f'results/{model.__class__.__name__}_acc.png')
        
    return model

def evaluate_model(model, test_loader, task='classification'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device).float(), y.to(device).float()
            out = model(X)
            all_preds.append(out.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    return np.vstack(all_preds), np.vstack(all_targets)
