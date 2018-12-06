import torch
import torch.nn as nn

from torch_localize import localized_module

from .cmplx import magnitude, magnitude_sqr
from .multidim import std, mean

class _Normalization(nn.Module):
    def __init__(self, repr, dim=2, eps=1e-3, kind='batch'):
        super(_Normalization, self).__init__()

        if dim not in [2, 3]:
            raise ValueError(f"Allowed dim values are 2 and 3, got {dim}")

        kinds = ['batch', 'instance', 'item']
        if kind not in kinds:
            msg = f"Allowed kind values are '{kinds}', got {kind}"
            raise ValueError(msg)

        self.repr = repr
        self.eps = eps
        self.n_features = sum(repr)

        self._dim = dim
        self._kind = kind

    def forward(self, x: [2, 'b', 'f', 'w', 'h', ...]) -> [2, 'b', 'f', 'w', 'h', ...]:
        if x.shape[2] != self.n_features:
            fmt = ("1st dimension on input `x` ({}) doesn't match with "
                   "declared # of features ({} == sum({}))")
            msg = fmt.format(x.shape[2], self.n_features, tuple(self.repr))
            raise ValueError(msg)

        reduce_dims = [3, 4]
        if self._kind == 'batch':
            reduce_dims.append(1)
        if self._kind == 'item':
            reduce_dims.append(2)
        if self._dim == 3:
            reduce_dims.append(5)

        means = mean(x, dim=reduce_dims, keepdim=True)
        x = x - means

        stds = std(magnitude(x), dim=[d-1 for d in reduce_dims], keepdim=True)
        x = x / (stds.unsqueeze(0) + self.eps)

        return x

    def __repr__(self):
        fmt = '{}Norm{}d(repr={}, eps={})'
        msg = fmt.format(self._kind.capitalize(), self._dim, self.repr, self.eps)
        return msg
