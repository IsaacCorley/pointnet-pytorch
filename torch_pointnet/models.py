import torch
import torch.nn as nn
import torch.nn.functional as F


def ConvBlock(nn.Module):

    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv1d(),
            nn.BatchNorm1d(),
            nn.R
            

        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def MLP(nn.Module):

    def __init__(self):
        pass


class TNet(nn.Module):

    def __init__(self, k: int):
        self.k = k

class Transform(nn.Module):

    def __init__(self):
        self.tnet = TNet(k=3)
class PointNet(nn.Module):

    def __init__(self):
        pass