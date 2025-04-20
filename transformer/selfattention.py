import numpy as np
import torch
from torch import nn
import math


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class SelfAttention(nn.Module):
    def __init__(self, input_dims, num_hiddens, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.num_hiddens = num_hiddens
        self.w_q = nn.Linear(input_dims, num_hiddens)
        self.w_k = nn.Linear(input_dims, num_hiddens)
        self.w_v = nn.Linear(input_dims, num_hiddens)
        self.out_proj = nn.Linear(num_hiddens, input_dims)

        self.softmax = nn.Softmax(dim=- 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        attn = torch.matmul(q, k.transpose(-1, -2)) / \
            math.sqrt(self.num_hiddens)
        print(attn.shape)
        attn = self.softmax(attn)
        output = attn @ v
        out = self.out_proj(output)
        return out


class SelfAttentionv1(nn.Module):
    def __init__(self, input_dims, num_hiddens, dropout_rate=0.1):
        super().__init__()
        self.nums_hidden = num_hiddens
        self.w_q = nn.Linear(input_dims, num_hiddens)
        self.w_k = nn.Linear(input_dims, num_hiddens)
        self.w_v = nn.Linear(input_dims, num_hiddens)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(num_hiddens, input_dims)

    def forward(self, x, mask=None):

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        print(Q.shape)
        attn = Q @ K.transpose(-1, -2) / math.sqrt(self.nums_hidden)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        score = attn @ V
        out = self.output(score)

        return out


if __name__ == "__main__":
    x = torch.randn(2, 100, 24)
    mask = torch.eye(100).repeat(2, 1, 1)
    net = SelfAttentionv1(input_dims=24, num_hiddens=32)
    out = net(x, mask)
    print(out.shape)
    print(out)


class sa(nn.Module):
    def __init__(self, num_hiddens, dropout_rate=0.1):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.wq = nn.Linear(num_hiddens, num_hiddens)
        self.wk = nn.Linear(num_hiddens, num_hiddens)
        self.wv = nn.Linear(num_hiddens, num_hiddens)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(num_hiddens, num_hiddens)

    def forward(self, x, mask=None):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        attn = q @ k.transpose(-1, -2) / math.sqrt(self.num_hiddens)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        score = attn @ v
        out = self.output(score)
        return out


seq_len = 100
m = sa(32)
x = torch.randn(2, 100, 32)
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).repeat(2, 1, 1)
out = m(x, mask)
print(out.shape)
print(out)
