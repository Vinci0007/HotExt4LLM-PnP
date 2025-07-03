
import torch   
import torch.nn as nn   
import torch.nn.functional as F   

class PnPNystromAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_landmarks=64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_landmarks = num_landmarks
        self.head_dim = dim // num_heads

        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape  # Batch, Seq_len, Channels

        Q = self.Wq(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, N, d)
        K = self.Wk(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        idx = torch.linspace(0, N-1, self.num_landmarks).long().to(x.device)
        Q_landmarks = Q[:, :, idx, :]  # (B, h, m, d)
        K_landmarks = K[:, :, idx, :]  # (B, h, m, d)

        scale = self.head_dim ** 0.5
        GL = torch.exp(torch.matmul(Q_landmarks, K_landmarks.transpose(-2, -1)) / scale)  # (B, h, m, m)

        GU = torch.exp(torch.matmul(Q_landmarks, K.transpose(-2, -1)) / scale)  # (B, h, m, N)

        GQ = torch.exp(torch.matmul(Q, K_landmarks.transpose(-2, -1)) / scale)  # (B, h, N, m)

        GL_inv = torch.linalg.pinv(GL)  # (B, h, m, m)

        G_hat = torch.matmul(torch.matmul(GQ, GL_inv), GU)  # (B, h, N, N)

        G_hat = G_hat / (G_hat.sum(-1, keepdim=True) + 1e-6)

        out = torch.matmul(G_hat, V)  # (B, h, N, d)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        return out