import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):
    def __init__(self, d_input, d_k=64, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_input = d_input
        self.d_k = d_k

        self.W_q = nn.Linear(d_input, d_k, bias=False)
        self.W_k = nn.Linear(d_input, d_k, bias=False)
        self.W_v = nn.Linear(d_input, d_k, bias=False)

    def phi(self, x):
        return F.elu(x) + 1  # positive feature map

    def forward(self, X):
        X = X.to(self.device)
        Q = self.phi(self.W_q(X))  # (n, d_k)
        K = self.phi(self.W_k(X))  # (n, d_k)
        V = self.W_v(X)            # (n, d_k)

        # Precompute key-value summary
        KV = K.T @ V               # (d_k, d_k)
        sum_K = K.sum(dim=0, keepdim=True)  # (1, d_k)

        # Normalization term
        Z = 1.0 / (Q @ sum_K.T).clamp(min=1e-6)  # (n, 1)

        # Linear attention output
        out = (Q @ KV) * Z
        return out
