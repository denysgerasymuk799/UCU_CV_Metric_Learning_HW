import torch

import torch.nn as nn
import torch.nn.functional as F


class CustomContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Find an euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        # Compute a contrastive loss
        contrastive_loss = torch.mean(label * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return contrastive_loss
