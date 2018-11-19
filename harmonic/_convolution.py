import torch, itertools, math
import torch.nn as nn
import torch.nn.functional as F
from torch_localize import localized_module
from torch_dimcheck import dimchecked

from .cmplx import cmplx 
from .weights import Weights

@dimchecked
def cconv_nd(x: [2, 'b',     'f_in', 'hx', 'wx', ...,],
             w: [2, 'f_out', 'f_in', 'hk', 'wk', ...,],
             dim=2, pad=False) -> [2, 'b', 'f_out', 'ho', 'wo', ...]:

    if dim not in [2, 3]:
        raise ValueError("Dim can only be 2 or 3, got {}".format(dim))

    if pad:
        padding = w.shape[3] // 2
    else:
        padding = 0

    conv = F.conv3d if dim == 3 else F.conv2d

    real = conv(x[0, ...], w[0, ...], padding=padding) - \
           conv(x[1, ...], w[1, ...], padding=padding)

    imag = conv(x[0, ...], w[1, ...], padding=padding) + \
           conv(x[1, ...], w[0, ...], padding=padding)

    return cmplx(real, imag)


def ords2s(in_ord, out_ord):
    return '{}_{}'.format(in_ord, out_ord)


class _HConv(nn.Module):
    def __init__(self, in_repr, out_repr, size, radius=None, dim=2, pad=False):
        super(_HConv, self).__init__()

        if dim not in [2, 3]:
            raise ValueError("Dim can only be 2 or 3, got {}".format(dim))

        self.dim = dim
        self.in_repr = in_repr
        self.out_repr = out_repr
        self.size = size
        self.pad = pad

        self.radius = radius if radius is not None else size / 2 - 1
        self.weights = nn.ModuleDict()

        # 2d convolutions take features_in feature maps as input,
        # 3d take features_in * size feature maps (features_in times
        # size of z-dimension)
        mul = 1 if dim == 2 else size

        # create Weights for each (input order, output order) pair
        for (in_ord, in_mult), (out_ord, out_mult) in itertools.product(
                                    enumerate(in_repr),
                                    enumerate(out_repr)):

            if in_mult == 0 or out_mult == 0:
                # either order is not represented in current (in, out) pair
                continue

            name = 'Weights {}x{} -> {}x{}'.format(
                in_mult, in_ord, out_mult, out_ord
            )


            order_diff = in_ord - out_ord
            weight = Weights(
                in_mult * mul, out_mult, size, self.radius, order_diff, name=name
            )
            self.weights[ords2s(in_ord, out_ord)] = weight 

    def __repr__(self):
        fmt = 'HConv{}d(repr_in={}, repr_out={}, size={}, radius={})'
        msg = fmt.format(
            self.dim, self.in_repr, self.out_repr, self.size, self.radius
        )
        return msg

    def synthesize(self) -> [2, 'fo', 'fi', 'h', 'w', ...]:
        spatial_unsqueeze = [self.size] * self.dim

        input_kernels = []
        for (in_ord, in_mult) in enumerate(self.in_repr):
            output_kernels = []
            if in_mult == 0:
                continue

            for (out_ord, out_mult) in enumerate(self.out_repr):
                if out_mult == 0:
                    continue

                ix = ords2s(in_ord, out_ord)
                kernel = self.weights[ix].cartesian_harmonics()
                kernel = kernel.reshape(
                    2, out_mult, in_mult, *spatial_unsqueeze
                )

                output_kernels.append(kernel)

            input_kernels.append(torch.cat(output_kernels, dim=1))

        return torch.cat(input_kernels, dim=2)
        
    def forward(self, x: [2, 'b', 'fi', 'hx', 'wx', ...]
               ) -> [2, 'b', 'fo', 'ho', 'wo', ...]:
        if x.shape[2] != sum(self.in_repr):
            fmt = "Based on repr {} expected {} feature maps, found {}"
            msg = fmt.format(self.in_repr, sum(self.in_repr), x.shape[2])
            raise ValueError(msg)

        return cconv_nd(x, self.synthesize(), dim=self.dim, pad=self.pad)
