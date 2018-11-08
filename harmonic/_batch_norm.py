import torch
import torch.nn as nn

from torch_localize import localized_module

from .cmplx import magnitude

class _BatchNorm(nn.Module):
    def __init__(self, repr, dim=2, eps=1e-2):
        super(_BatchNorm, self).__init__()

        if dim not in [2, 3]:
            raise ValueError("Allowed dim values are 2 and 3, got {}".format(dim))

        self.dim = dim
        self.repr = repr
        self.eps = eps
        self.n_features = sum(repr)


    def forward(self, x: ['b', 'f', 'w', 'h', ..., 2]) -> ['b', 'f', 'w', 'h', ..., 2]:
        if x.shape[1] != self.n_features:
            fmt = ("1st dimension on input `x` ({}) doesn't match with "
                   "declared # of features ({} == sum({}))")
            msg = fmt.format(x.shape[1], self.n_features, tuple(self.repr))
            raise ValueError(msg)
        flat = x.transpose(0, 1).reshape(self.n_features, -1, 2)

        # compute mean
        means = flat.mean(dim=1, keepdim=True)

        # compute std
        stds = magnitude(flat - means).std(dim=1) + self.eps

        spatial = [1] * self.dim
        mean_corrected = x - means.reshape(1, -1, *spatial, 2)
        std_corrected = mean_corrected / stds.reshape(1, -1, *spatial, 1)


        return std_corrected
