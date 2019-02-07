import torch
import torch.nn as nn

from torch_localize import localized_module

from ..cmplx import magnitude, magnitude_sqr
from ..multidim import std, mean

class RayleighNorm2d(nn.Module):
    def __init__(self, repr, eps=1e-3):
        super(RayleighNorm2d, self).__init__()

        self.repr = repr
        self.n_features = sum(repr)
        self.eps = eps

    def forward(self, x: [2, 'b', 'f', 'w', 'h']) -> [2, 'b', 'f', 'w', 'h']:
        if x.shape[2] != self.n_features:
            fmt = ("1st dimension on input `x` ({}) doesn't match with "
                   "declared # of features ({} == sum({}))")
            msg = fmt.format(x.shape[2], self.n_features, tuple(self.repr))
            raise ValueError(msg)

        mean_ = mean(x, dim=[3, 4], keepdim=True)
        magn_sqr = magnitude_sqr(x - mean_)
        sigma = mean(magn_sqr, dim=[2, 3], keepdim=True) / 2.

        return x / (sigma.sqrt().unsqueeze(0) + self.eps)

    def __repr__(self):
        fmt = 'RayleighNorm2d(repr={}, eps={})'
        msg = fmt.format(self.repr, self.eps)
        return msg
