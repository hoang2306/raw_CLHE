
import torch
import torch.nn as nn 
import torch.nn.functional as F


class LinearAttention(nn.Module):
    def __init__(self, d_input, d_k=64, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_input = d_input
        self.d_k = d_k

        self.W_q = nn.Parameter(torch.randn(d_input, d_k))
        self.W_k = nn.Parameter(torch.randn(d_input, d_k))
        self.W_v = nn.Parameter(torch.randn(d_input, d_k))

    def phi(self, x):
        return F.elu(x) + 1

    def forward(self, X):
        X = X.to(self.device)
        n, d = X.shape

        Q = X @ self.W_q   # (n, d_k)
        K = X @ self.W_k   # (n, d_k)
        V = X @ self.W_v   # (n, d_k)

        Q_phi = self.phi(Q)
        K_phi = self.phi(K)

        # Compute linear attention
        KV = K_phi.T @ V # (d_k, d_k)
        ones_vec = torch.ones(n, 1, device=self.device)
        Z = 1 / (Q_phi @ K_phi.T @ ones_vec + 1e-8)   # add eps for stability
        out = (Q_phi @ KV) * Z # (n, d_k)

        return out