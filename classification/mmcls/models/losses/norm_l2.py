import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MSE_Norm_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return F.mse_loss(x, y)