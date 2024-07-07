import torch
from torch import nn
import torch.nn.functional as F



class MPA(nn.Module):
  
    def __init__(self, value: float = 1e-6) :
        super().__init__()
        self.value = value

    def forward(self, inputs, anwer) :
        with torch.no_grad():
            inputs = inputs.softmax(-3).argmax(-3, keepdim=True)

            prediction1 = ~inputs.bool()
            predictions2 = inputs.bool()
            inputs = torch.cat([prediction1, predictions2], dim=-3)

            anwer = anwer.bool()

            union = (inputs & anwer).sum((-2, -1))
            total = anwer.sum((-2, -1))

            mpa = (union + self.value) / (total + self.value)
            return torch.mean(mpa)