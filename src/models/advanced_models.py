import torch
import torch.nn as torch_nn
import torch.nn.functional as F
import math

class PositionalEncoding(torch_nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerENose(torch_nn.Module):
    def __init__(self, in_channels=6, d_model=64, nhead=4, dim_feedforward=256, output_dim=4, task='classification'):
        super().__init__()
        self.task = task
        self.in_channels = in_channels
        self.input_linear = torch_nn.Linear(in_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = torch_nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = torch_nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc = torch_nn.Linear(d_model, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            # (batch, 126) -> (batch, 21, 6)
            x = x.view(x.size(0), -1, self.in_channels)
        elif x.dim() == 3 and x.size(1) == self.in_channels:
            x = x.transpose(1, 2)
            
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        out = self.fc(x)
        return out

class GraphConvLayer(torch_nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch_nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch_nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x, adj):
        return torch.relu(torch.matmul(adj, torch.matmul(x, self.weight)))

class GNN(torch_nn.Module):
    def __init__(self, in_features=21, hidden_features=64, output_dim=4, num_nodes=6, task='classification'):
        super().__init__()
        self.task = task
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.gc1 = GraphConvLayer(in_features, hidden_features)
        self.gc2 = GraphConvLayer(hidden_features, 32)
        self.fc = torch_nn.Linear(32, output_dim)

    def forward(self, x, adj=None):
        if x.dim() == 2:
            # (batch, 126) -> (batch, 6, 21)
            x = x.view(x.size(0), self.num_nodes, -1)
        if adj is None:
            adj = torch.ones(self.num_nodes, self.num_nodes, device=x.device)
            adj = adj / self.num_nodes
            
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out

class PIML_Loss(torch_nn.Module):
    def __init__(self, task='regression', lambda_1=0.1, lambda_2=0.01):
        super().__init__()
        self.task = task
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.data_loss = torch_nn.MSELoss() if task == 'regression' else torch_nn.BCEWithLogitsLoss()

    def forward(self, preds, targets, inputs=None):
        loss = self.data_loss(preds, targets)
        
        if inputs is not None:
            if inputs.dim() == 2:
                inputs_3d = inputs.view(inputs.size(0), -1, 6)
            else:
                inputs_3d = inputs
            diffs = inputs_3d[:, 1:, :] - inputs_3d[:, :-1, :]
            l_mono = torch.mean(torch.relu(-diffs))
            second_diffs = diffs[:, 1:, :] - diffs[:, :-1, :]
            l_smooth = torch.mean(second_diffs ** 2)
            loss += self.lambda_1 * l_mono + self.lambda_2 * l_smooth
            
        return loss

class PIML_Model(torch_nn.Module):
    def __init__(self, input_dim, output_dim, task='classification'):
        super().__init__()
        self.task = task
        self.net = torch_nn.Sequential(
            torch_nn.Linear(input_dim, 128),
            torch_nn.BatchNorm1d(128),
            torch_nn.ReLU(),
            torch_nn.Dropout(0.2),
            torch_nn.Linear(128, 64),
            torch_nn.BatchNorm1d(64),
            torch_nn.ReLU(),
            torch_nn.Linear(64, output_dim)
        )

    def forward(self, x):
        if x.dim() > 2:
            x_flat = x.view(x.size(0), -1)
        else:
            x_flat = x
            
        out = self.net(x_flat)
        return out
