import torch
import torch.nn as nn
from torch_dimcheck import ShapeChecker

from .cmplx import magnitude

class _ScalarGate(nn.Module):
    def __init__(self, repr, dim=2, mult=2):
        super(_ScalarGate, self).__init__()

        if dim not in [2, 3]:
            raise ValueError("Dim value of {} is not allowed".format(dim))

        self.repr = repr
        self.dim = dim
        self.mult = mult

        conv = nn.Conv2d if dim == 2 else nn.Conv3d

        total_fmaps = sum(repr)

        if mult == 2:
            self.seq = nn.Sequential(
                conv(total_fmaps, total_fmaps, 1),
                nn.ReLU(inplace=True),
                conv(total_fmaps, total_fmaps, 1),
            )
        elif mult == 1:
            self.seq = nn.Sequential(
                conv(total_fmaps, total_fmaps, 1)
            )
        else:
            raise ValueError(f"mult={mult} is not supported")

    def forward(self, x: [2, 'b', 'f', 'h', 'w', ...]) -> [2, 'b', 'f', 'h', 'w', ...]:
        magnitudes = magnitude(x)
        
        g = self.seq(magnitudes)
        g = torch.sigmoid(g)

        return x * g.unsqueeze(0)

    def __repr__(self):
        fmt = 'ScalarGate{}d(repr={}, mult={})'
        msg = fmt.format(self.dim, self.repr, self.mult)
        return msg
