import torch
import torch.nn as nn

from torch_localize import localized_module

from .cmplx import magnitude, magnitude_sqr

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

        # transpose such that we can reduce over batches (in case of batch
        # norm) while keeping feature channels separate, then flatten
        # the spatial + (in case of batch norm) batch dimensions
        flat = x.transpose(1, 2).reshape(2, self.n_features * batch_dim, -1)

        # compute mean and std
        means = flat.mean(dim=2, keepdim=True)
        mean_corrected = flat - means

        stds = magnitude(mean_corrected).std(dim=1, keepdim=True)
        std_corrected = mean_corrected / stds.unsqueeze(0)

        # recover the transposed shape then transpose back into the original shape
        transposed_shape = (2, x.shape[2], x.shape[1], *x.shape[3:])
        return std_corrected.reshape(*transposed_shape).transpose(1, 2)

    def __repr__(self):
        fmt = '{}Norm{}d(repr={}, eps={})'
        msg = fmt.format(self._kind.capitalize(), self._dim, self.repr, self.eps)
        return msg
