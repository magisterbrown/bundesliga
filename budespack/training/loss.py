import torch.nn as nn
import torch

def resp_loss(x, y):
    seq_len = x.shape[-1]
    loss = nn.BCEWithLogitsLoss()
    y = y.view(-1, seq_len)
    return loss(x, y)
