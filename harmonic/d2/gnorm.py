import torch
import torch.nn as nn
from torch_dimcheck import dimchecked

from ..cmplx import magnitude, magnitude_sqr
from ..multidim import std, mean

class GroupNorm2d(nn.Module):
    def __init__(self, repr, eps=1e-3):
        super(GroupNorm2d, self).__init__()

        self.repr = repr
        self.eps = eps
        self.n_features = sum(repr)

    def forward(self, x: [2, 'b', 'f', 'w', 'h']) -> [2, 'b', 'f', 'w', 'h']:
        if x.shape[2] != self.n_features:
            fmt = ("1st dimension on input `x` ({}) doesn't match with "
                   "declared # of features ({} == sum({}))")
            msg = fmt.format(x.shape[2], self.n_features, tuple(self.repr))
            raise ValueError(msg)

        groups = []
        prev = 0
        for mult in self.repr:
            group = x[:, :, prev:prev+mult, ...]
            means = mean(x, dim=(1, 2, 3, 4), keepdim=True)
            group = group - means

            stds = std(magnitude(group), dim=(0, 1, 2, 3), keepdim=True)
            group = group / (stds.unsqueeze(0) + self.eps)

            groups.append(group)

        return torch.cat(groups, dim=2)

    def __repr__(self):
        fmt = 'GroupNorm2d(repr={}, eps={})'
        msg = fmt.format(self.repr, self.eps)
        return msg
