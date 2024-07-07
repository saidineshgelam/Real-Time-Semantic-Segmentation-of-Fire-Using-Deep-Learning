import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, a_l: float = 0.25, g_l: float = 2.) -> None:
        super().__init__()
        self.a_l = a_l
        self.g_l = g_l

    def forward(self, value, target) :
        classifcation_loss = F.binary_cross_entropy_with_logits(
            value, target, reduction='none')
        pt = torch.exp(-classifcation_loss)
        loss = self.a_l * (1 - pt) ** self.g_l * classifcation_loss
        return loss.mean()