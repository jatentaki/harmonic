import torch
import torch.nn as nn
from torch_dimcheck import ShapeChecker

from .cmplx import magnitude

class _ScalarGate(nn.Module):
    def __init__(self, repr, dim=2):
        super(_ScalarGate, self).__init__()

        if dim not in [2, 3]:
            raise ValueError("Dim value of {} is not allowed".format(dim))

        self.repr = repr
        self.dim = dim

        conv = nn.Conv2d if dim == 2 else nn.Conv3d

        total_fmaps = sum(repr)
        self.conv1 = conv(total_fmaps, total_fmaps, 1)
        self.conv2 = conv(total_fmaps, total_fmaps, 1)

    def forward(self, x: [2, 'b', 'f', 'h', 'w', ...]) -> [2, 'b', 'f', 'h', 'w', ...]:
        magnitudes = magnitude(x)
        
        g = self.conv1(magnitudes)
        g = torch.relu(g)
        g = self.conv2(g)
        g = torch.sigmoid(g)

        return x * g.unsqueeze(0)
