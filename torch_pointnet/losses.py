import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetLoss(nn.Module):

    def __init__(self, weight_decay: float = 0.0001):
        super().__init__()
        self.weight_decay = weight_decay
        self.loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.MSELoss()

    def forward(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        input_matrix: torch.Tensor,
        feature_matrix: torch.Tensor
    ) -> torch.Tensor:

        batch_size = y_pred.size(0)

        # Orthogonal regularization loss
        input_identity = torch.eye(input_matrix.size(-1), requires_grad=True).repeat(batch_size, 1, 1)
        feature_identity = torch.eye(feature_matrix.size(-1), requires_grad=True).repeat(batch_size, 1, 1)
        
        orthogonal_input_loss = self.reg_loss(
            torch.bmm(input_matrix, input_matrix.transpose(1, 2)),
            input_identity
        )
        orthogonal_feature_loss = self.reg_loss(
            torch.bmm(feature_matrix, feature_matrix.transpose(1, 2)),
            feature_identity
        )
        orthogonal_reg_loss = (orthogonal_input_loss + orthogonal_feature_loss) / float(batch_size)
        
        # Classification loss
        class_loss = self.loss(y_pred, y)

        return class_loss + self.weight_decay * orthogonal_reg_loss
