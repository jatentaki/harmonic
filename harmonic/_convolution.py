import torch, itertools, math
import torch.nn as nn
import torch.nn.functional as F
from torch_localize import localized_module
from torch_dimcheck import dimchecked, ShapeChecker

from .cmplx import cmplx 
from .weights import Weights

@dimchecked
def cconv_nd(x: ['b',     'f_in', 'hx', 'wx', ..., 2],
                 w: ['f_out', 'f_in', 'hk', 'wk', ..., 2],
                 dim=2, pad=False) -> ['b', 'f_out', 'ho', 'wo', ..., 2]:

    if dim not in [2, 3]:
        raise ValueError("Dim can only be 2 or 3, got {}".format(dim))

    if pad:
        padding = w.shape[3] // 2
    else:
        padding = 0

    conv = F.conv3d if dim == 3 else F.conv2d

    real = conv(x[..., 0], w[..., 0], padding=padding) - \
           conv(x[..., 1], w[..., 1], padding=padding)

    imag = conv(x[..., 0], w[..., 1], padding=padding) + \
           conv(x[..., 1], w[..., 0], padding=padding)

    return cmplx(real, imag)

@localized_module
class _StreamHConv(nn.Module):
    def __init__(self, in_channels, out_channels, size, order,
                 radius=None, pad=False, dim=2):
        super(_StreamHConv, self).__init__()

        if dim not in [2, 3]:
            raise ValueError("Dim can only be 2 or 3, got {}".format(dim))
        else:
            self.dim = dim

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size
        self.order = order

        self.radius = radius if radius is not None else size / 2 - 1
        self.pad = pad

        n_in = size * in_channels if dim == 3 else in_channels
        self.weights = Weights(n_in, out_channels, size, self.radius, order)

    @dimchecked
    def forward(self, t: ['b', 'fi', 'hi', 'wi', ..., 2]) -> ['b', 'fo', 'ho', 'wo', ..., 2]:

        kernel = self.weights.cartesian_harmonics()
        kernel = kernel.reshape(
            self.out_channels, self.in_channels, *([self.size] * self.dim), 2
        )

        return cconv_nd(t, kernel, pad=self.pad, dim=self.dim)


def ords2s(in_ord, out_ord):
    return '{}_{}'.format(in_ord, out_ord)


class _HConv(nn.Module):
    def __init__(self, in_repr, out_repr, radius, dim=2, pad=False):
        super(_HConv, self).__init__()

        if dim not in [2, 3]:
            raise ValueError("Dim can only be 2 or 3, got {}".format(dim))
        else:
            self.dim = dim

        self.in_repr = in_repr
        self.out_repr = out_repr

        # input/output shapes depending on dimensionality
        if dim == 2:
            self.in_dims = ['n', -1, 'hi', 'wi', 2]
            self.out_dims = ['n', -1, 'ho', 'wo', 2]
        else:
            self.in_dims = ['n', -1, 'hi', 'wi', 'di', 2]
            self.out_dims = ['n', -1, 'ho', 'wo', 'do', 2]

        self.convs = nn.ModuleDict()

        # create an HConv which maps between all pairs on (input, output) streams
        for (in_ord, in_mult), (out_ord, out_mult) in itertools.product(
                                    enumerate(in_repr),
                                    enumerate(out_repr)):

            if in_mult == 0 or out_mult == 0:
                # either order is not represented in current (in, out) pair
                continue

            name = 'StreamHConv{} {}x{} -> {}x{}'.format(
                dim, in_mult, in_ord, out_mult, out_ord
            )
            conv = _StreamHConv(
                in_mult, out_mult, radius, in_ord - out_ord, pad=pad, name=name, dim=dim
            )
            self.convs[ords2s(in_ord, out_ord)] = conv

    def forward(self, *streams):
        if len(streams) != len(self.in_repr):
            fmt = "Based on repr {} expected {} streams, got {}"
            msg = fmt.format(self.in_repr, len(self.in_repr), len(streams))
            raise ValueError(msg)

        checker = ShapeChecker()
        for i, stream in enumerate(streams):
            if stream is None:
                continue

            checker.check(stream, self.in_dims, name='in_stream{}'.format(i))

        out_streams = [(0 if repr != 0 else None) for repr in self.out_repr]

        for in_ord, in_stream in enumerate(streams):
            if in_stream is None:
                continue

            for out_ord in range(len(out_streams)):
                if out_streams[out_ord] is None:
                    continue

                conv = self.convs[ords2s(in_ord, out_ord)]
                out_streams[out_ord] += conv(in_stream)

        for i, stream in enumerate(out_streams):
            if stream is None:
                continue

            checker.check(stream, self.out_dims, name='out_stream{}'.format(i))

        return out_streams
