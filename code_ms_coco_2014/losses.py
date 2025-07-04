import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC

ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE

        return focal_loss


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class ContraLoss(nn.Module):

    def __init__(self, temperature = 0.3, weight=None, size_average=True):
        super().__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, embedd_x: torch.Tensor, embedd_y: torch.Tensor, mask_x: torch.Tensor, mask_y: torch.Tensor):
        x_embedding = self.norm_embed(embedd_x)
        y_embedding = self.norm_embed(embedd_y)

        x_masks = F.interpolate(mask_x, size=x_embedding.shape[-2:], mode="bilinear", align_corners=False).detach()

        sum_x = x_masks.sum(dim=[-1, -2]).clone()

        y_masks = F.interpolate(mask_y, size=y_embedding.shape[-2:], mode="bilinear", align_corners=False).detach()
        sum_y = y_masks.sum(dim=[-1, -2]).clone()

        multi_embedd_x = (x_embedding * x_masks).sum(dim=[-1, -2]) / sum_x  # [n, 256, 64, 64] >> [n, 256]
        multi_embedd_y = (y_embedding * y_masks).sum(dim=[-1, -2]) / sum_y

        flatten_x = multi_embedd_x.view(multi_embedd_x.size(0), -1)
        flatten_y = multi_embedd_y.view(multi_embedd_y.size(0), -1)
        similarity_matrix = F.cosine_similarity(flatten_x.unsqueeze(1), flatten_y.unsqueeze(0), dim=2)

        label_pos = torch.eye(x_masks.size(0)).bool().to(embedd_x.device)
        label_nag = ~label_pos

        similarity_matrix = similarity_matrix / self.temperature    # [n insts, n insts]
        loss = -torch.log(
                similarity_matrix.masked_select(label_pos).exp().sum() / 
                similarity_matrix.exp().sum()
            )

        return loss

    def norm_embed(self, embedding: torch.Tensor):
        embedding = F.normalize(embedding, dim=0, p=2)
        return embedding

