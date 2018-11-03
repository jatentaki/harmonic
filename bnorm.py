import torch
import torch.nn as nn

from torch_dimcheck import dimchecked

from cmplx import magnitude

class BatchNorm2d(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super(BatchNorm2d, self).__init__()

        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_vars', torch.ones(num_features))

    @dimchecked
    def forward(self, x: ['b', 'f', 'w', 'h', 2]) -> ['b', 'f', 'w', 'h', 2]:
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

        norms = norms.reshape(1, -1, 1, 1, 1)

        return x / norms

