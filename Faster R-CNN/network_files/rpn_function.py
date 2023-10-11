import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Dict, Tuple

class AnchorsGenerator(nn.Module):
    pass


class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors) -> None:
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                              kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors*4, kernel_size=1, stride=1, padding=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        logits = []
        bbox_reg = []
        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg
     