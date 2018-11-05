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
        self.subnorms = nn.ModuleList()

        for i, mult in enumerate(repr):
            if mult == 0:
                bnorm = None
            else:
                name = 'bn{}d_{}'.format(dim, i)
                bnorm = BatchNorm(mult, momentum=momentum, eps=eps, name=name, dim=dim)

            self.subnorms.append(bnorm)

    def forward(self, *streams):
        if len(streams) != len(self.repr):
            fmt = "Based on repr {} expected {} streams, got {}"
            msg = fmt.format(self.repr, len(self.repr), len(streams))
            raise ValueError(msg)

        out_streams = []
        for i, (stream, bnorm) in enumerate(zip(streams, self.subnorms)):
            if stream is None and bnorm is not None:
                fmt = "Stream {} has no channels, but was expected to have {}"
                msg = fmt.format(i, self.repr[i])
                raise ValueError(msg)
            
            if stream is not None and bnorm is None:
                fmt = "Stream {} has {} channels, but was expected to have none"
                msg = fmt.format(i, stream.shape[1])
                raise ValueError(msg)

            if stream is None:
                out_streams.append(None)
            else:
                out_streams.append(bnorm(stream))

        return out_streams


@localized_module
class _StreamBatchNorm(nn.Module):
    def __init__(self, num_features, dim=3, momentum=0.1, eps=1e-5):
        super(_StreamBatchNorm, self).__init__()

        if dim not in [2, 3]:
            raise ValueError("Allowed dim values are 2 and 3, got {}".format(dim))
        self.dim = dim

        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_vars', torch.ones(num_features))

    def forward(self, x):
        if x.shape[1] != self.num_features:
            fmt = ("1st dimension on input `x` ({}) doesn't match with "
                   "declared # of features ({})")
            msg = fmt.format(x.shape[1], self.num_features)
            raise ValueError(msg)

        if self.eval:
            norms = self.running_vars + self.eps
        else:
            magnitudes = magnitude(x)

            # transpose to get shape ['f', ...], then flatten batch and spatial dims
            magnitudes = magnitudes.transpose(0, 1).reshape(self.num_features, -1)
            stds = magnitudes.std(dim=1)

            norms = stds + self.eps

            new_vars = self.momentum * stds + (1 - self.momentum) * self.running_vars
            self.running_vars = new_vars.detach()

        norms = norms.reshape(1, -1, *([1] * self.dim), 1)

        return x / norms
