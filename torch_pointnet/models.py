from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        bn: bool = True,
        activation: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            )
        ]

        if bn:
            layers.append(nn.BatchNorm1d(out_channels))

        if activation:
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FCBlock(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bn: bool = True,
        activation: bool = True,
        p: float = 0.0,
    ):
        super().__init__()
        layers = [
            nn.Linear(
                in_features=in_features,
                out_features=out_features,
            )
        ]

        if p != 0.0:
            layers.append(nn.Dropout(p))

        if bn:
            layers.append(nn.BatchNorm1d(out_features))

        if activation:
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TNet(nn.Module):

    def __init__(self, k: int, n: int):
        super().__init__()
        self.k = k
        self.n = n
        self.model = nn.Sequential(
            ConvBlock(in_channels=k, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=1024),
            nn.MaxPool1d(kernel_size=n),
            nn.Flatten(),
            FCBlock(in_features=1024, out_features=512),
            FCBlock(in_features=512, out_features=256),
            FCBlock(in_features=256, out_features=k*k, bn=False, activation=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        identity = torch.eye(self.k, requires_grad=True).repeat(x.size(0), 1, 1)
        identity = identity.to(x.device).to(x.dtype)
        matrix = x.view(-1, self.k, self.k) + identity
        return matrix

class Backbone(nn.Module):

    def __init__(self, n: int):
        super().__init__()
        self.input_transform = TNet(k=3, n=n)
        self.conv = ConvBlock(in_channels=3, out_channels=64)
        self.feature_transform = TNet(k=64, n=n)
        self.tail = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=1024, activation=False),
            nn.MaxPool1d(kernel_size=n),
            nn.Flatten()
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Input transform
        input_matrix = self.input_transform(x)
        x = torch.bmm(torch.transpose(x, 1, 2), input_matrix).transpose(1, 2)
        
        # Shared MLP
        x = self.conv(x)

        # Feature transform
        feature_matrix = self.feature_transform(x)
        local_features = torch.bmm(torch.transpose(x, 1, 2), feature_matrix).transpose(1, 2)

        # Shared MLP + pooling
        global_features = self.tail(local_features)

        return global_features, local_features, input_matrix, feature_matrix


class PointNetClassifier(nn.Module):

    def __init__(self, num_points: int, num_classes: int):
        super().__init__()
        self.backbone = Backbone(n=num_points)
        self.model = nn.Sequential(
            FCBlock(in_features=1024, out_features=512),
            FCBlock(in_features=512, out_features=256, p=0.3),
            FCBlock(
                in_features=256, out_features=num_classes, bn=False, activation=False
            ),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_features, local_features, input_matrix, feature_matrix = self.backbone(x)
        y_pred = self.model(global_features)
        return y_pred, input_matrix, feature_matrix


class PointNetSegmentation(nn.Module):

    def __init__(self, num_points: int, num_classes: int, classifier: PointNetClassifier):
        super().__init__()
        self.num_points = num_points
        self.num_classes = num_classes
        self.backbone = classifier.backbone
        self.model = nn.Sequential(
            ConvBlock(in_channels=1088, out_channels=512),
            ConvBlock(in_channels=512, out_channels=256),
            ConvBlock(in_channels=256, out_channels=128),
            ConvBlock(in_channels=128, out_channels=128),
            ConvBlock(
                in_channels=128, out_channels=num_classes, bn=False, activation=False
            )
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_features, local_features, input_matrix, feature_matrix = self.backbone(x)
        global_features = global_features.repeat(self.num_points, 1, 1)
        features = torch.cat((local_features, global_features))
        y_pred = self.model(features)
        return y_pred, input_matrix, feature_matrix
