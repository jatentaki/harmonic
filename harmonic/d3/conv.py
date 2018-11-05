import torch, itertools, math
import torch.nn as nn
import torch.nn.functional as F
from torch_localize import localized_module
from torch_dimcheck import dimchecked, ShapeChecker

from ..cmplx import cmplx 
from ..weights import Weights

@dimchecked
def complex_conv(x: ['b',     'f_in', 'hx', 'wx', 'dx', 2],
                 w: ['f_out', 'f_in', 'hk', 'wk', 'dk', 2],
                 pad=False) -> ['b', 'f_out', 'ho', 'wo', 'do', 2]:

    if pad:
        padding = w.shape[3] // 2
    else:
        padding = 0

    real = F.conv3d(x[..., 0], w[..., 0], padding=padding) - \
           F.conv3d(x[..., 1], w[..., 1], padding=padding)

    imag = F.conv3d(x[..., 0], w[..., 1], padding=padding) + \
           F.conv3d(x[..., 1], w[..., 0], padding=padding)

    return cmplx(real, imag)

@localized_module
class HConv(nn.Module):
    def __init__(self, in_channels, out_channels, size, order,
                 radius=None, pad=False):
        super(HConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size
        self.order = order

        self.radius = radius if radius is not None else size / 2 - 1
        self.pad = pad

        self.weights = Weights(size * in_channels, out_channels, size, self.radius, order)

    @dimchecked
    def forward(self, t: ['b', 'fi', 'hi', 'wi', 'di', 2]) -> ['b', 'fo', 'ho', 'wo', 'do', 2]:
        kernel = self.weights.cartesian_harmonics()
        kernel = kernel.reshape(
            self.out_channels, self.in_channels, self.size, self.size, self.size, 2
        )

        return complex_conv(t, kernel, pad=self.pad)


def ords2s(in_ord, out_ord):
    return '{}_{}'.format(in_ord, out_ord)


class CrossConv(nn.Module):
    def __init__(self, in_repr, out_repr, radius, pad=False):
        super(CrossConv, self).__init__()

        self.in_repr = in_repr
        self.out_repr = out_repr

        self.convs = nn.ModuleDict()

        # create an HConv which maps between all pairs on (input, output) streams
        for (in_ord, in_mult), (out_ord, out_mult) in itertools.product(
                                    enumerate(in_repr),
                                    enumerate(out_repr)):

            if in_mult == 0 or out_mult == 0:
                # either order is not represented in current (in, out) pair
                continue

            name = 'HConv {}x{} -> {}x{}'.format(in_mult, in_ord, out_mult, out_ord)
            conv = HConv(in_mult, out_mult, radius, in_ord - out_ord, pad=pad, name=name)
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

            checker.check(stream, ['n', -1, 'hi', 'wi', 'di', 2], name='in_stream{}'.format(i))

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

            checker.check(stream, ['n', -1, 'ho', 'wo', 'do', 2], name='out_stream{}'.format(i))

        return out_streams
