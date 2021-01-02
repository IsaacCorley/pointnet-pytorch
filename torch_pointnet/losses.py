import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.class_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.MSELoss()

    def forward(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        tmat: torch.Tensor
    ) -> torch.Tensor:
        pass