import torch
import torch.nn as nn

from torch_localize import localized_module

from .cmplx import magnitude

class _Normalization(nn.Module):
    def __init__(self, repr, dim=2, eps=1e-2, kind='batch'):
        super(_Normalization, self).__init__()

        if dim not in [2, 3]:
            raise ValueError(f"Allowed dim values are 2 and 3, got {dim}")

        if kind not in ['batch', 'instance']:
            msg = f"Allowed kind values are 'batch' and 'instance', got {kind}"
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

        if self._kind == 'batch':
            batch_dim = 1
        elif self._kind == 'instance':
            batch_dim = x.shape[1]

        flat = x.transpose(1, 2).reshape(2, self.n_features * batch_dim, -1)

        # compute mean
        means = flat.mean(dim=2, keepdim=True)

        # compute std
        stds = magnitude(flat - means).std(dim=1) + self.eps

        spatial = [1] * self._dim

        mean_corrected = x - means.reshape(2, batch_dim, -1, *spatial)
        std_corrected = mean_corrected / stds.reshape(1, batch_dim, -1, *spatial)

        return std_corrected

    def __repr__(self):
        fmt = '{}Norm{}d(repr={}, eps={})'
        msg = fmt.format(self._kind.capitalize(), self._dim, self.repr, self.eps)
        return msg
