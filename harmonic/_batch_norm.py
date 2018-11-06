import torch
import torch.nn as nn

from torch_localize import localized_module

from .cmplx import magnitude

class _BatchNorm(nn.Module):
    def __init__(self, repr, dim=2, momentum=0.1, eps=1e-5):
        super(_BatchNorm, self).__init__()

        if dim not in [2, 3]:
            raise ValueError("Allowed dim values are 2 and 3, got {}".format(dim))

        self.dim = dim
        self.repr = repr
        self.eps = eps
        self.momentum = momentum
        self.total_features = sum(repr)
        self.register_buffer('running_vars', torch.ones(self.total_features))


    def forward(self, x: ['b', 'f', 'w', 'h', ..., 2]) -> ['b', 'f', 'w', 'h', ..., 2]:
        if x.shape[1] != self.total_features:
            fmt = ("1st dimension on input `x` ({}) doesn't match with "
                   "declared # of features ({} == sum({}))")
            msg = fmt.format(x.shape[1], self.num_features, tuple(self.repr))
            raise ValueError(msg)

        if self.eval:
            norms = self.running_vars + self.eps
        else:
            magnitudes = magnitude(x)

            # transpose to get shape ['f', ...], then flatten batch and spatial dims
            magnitudes = magnitudes.transpose(0, 1).reshape(self.total_features, -1)
            stds = magnitudes.std(dim=1)

            norms = stds + self.eps

            new_vars = self.momentum * stds + (1 - self.momentum) * self.running_vars
            self.running_vars = new_vars.detach()

        norms = norms.reshape(1, -1, *([1] * self.dim), 1)

        return x / norms
