import torch, itertools, math
import torch.nn as nn
import torch.nn.functional as F
from torch_localize import localized_module
from torch_dimcheck import dimchecked, ShapeChecker

from cmplx import complex

@dimchecked
def h_conv(x: ['b',     'f_in', 'xh', 'xw', 2],
           w: ['f_out', 'f_in', 'kh', 'kw', 2],
           pad=False) -> ['b', 'f_out', 'oh', 'ow', 2]:

    if pad:
        padding = w.shape[3] // 2
    else:
        padding = 0

    real = F.conv2d(x[..., 0], w[..., 0], padding=padding) - \
           F.conv2d(x[..., 1], w[..., 1], padding=padding)

    imag = F.conv2d(x[..., 0], w[..., 1], padding=padding) + \
           F.conv2d(x[..., 1], w[..., 0], padding=padding)

    return complex(real, imag)

@localized_module
class HConv(nn.Module):
    def __init__(self, in_channels, out_channels, radius, order, pad=False):
        super(HConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.order = order
        self.pad = pad

        self.weights = Weights(in_channels, out_channels, radius, order)

    @dimchecked
    def forward(self, t: ['b', 'fi', 'hi', 'wi', 2]) -> ['b', 'fo', 'ho', 'wo', 2]:
        kernel = self.weights.synthesize()

        return h_conv(t, kernel, pad=self.pad)


class Weights(nn.Module):
    def __init__(self, in_channels, out_channels, radius, order, initialize=True):
        super(Weights, self).__init__()

        self.order = order
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.total_channels = in_channels * out_channels
        self.radius = radius
        self.diam = 2 * self.radius + 1

        self.r = nn.Parameter(
            torch.randn(self.total_channels, radius + 1, requires_grad=True)
        )
        self.betas = nn.Parameter(
            torch.randn(self.total_channels, 1, 1, requires_grad=True)
        )

        self.precompute_bilinear()
        self.precompute_angles()
        if initialize:
            self.initialize_weights()

    
    def precompute_bilinear(self):
        # compute radii on grid
        xs = torch.linspace(-self.radius, self.radius, self.diam)
        xs = xs.reshape(1, -1)
        ys = xs.reshape(-1, 1)
        rs = torch.sqrt(xs ** 2 + ys ** 2)

        # compute floor, ceil and alpha for bilinear interpolation
        floor = torch.floor(rs)
        self.register_buffer('alpha', (rs - floor).reshape(1, self.diam, self.diam))
        self.register_buffer('floor', floor.to(torch.int64))
        self.register_buffer('ceil', torch.ceil(rs).to(torch.int64))

        # in `radial` we extend the radial function with a 0 at the end such that we
        # can redirect out of bounds accesses there. Here we do it
        self.ceil[self.ceil > self.radius] = self.radius + 1
        self.floor[self.ceil > self.radius] = self.radius + 1


    def precompute_angles(self):
        # compute angles on grid
        xs = torch.linspace(-1, 1, self.diam)
        xs = xs.reshape(1, -1)
        ys = xs.reshape(-1, 1)
        self.register_buffer('angles', torch.atan2(xs, ys).unsqueeze(0))
    
    
    def initialize_weights(self):
        # we want to initialize such that assuming input from N(0, 1) the output
        # is as well in N(0, 1). This means each weight should also be from
        # Gaussian with mean 0 and sigma = 2 / sqrt(n_contributing_pixels).
        
        n_contributing = self.total_channels * self.diam ** 2
        std = 2. / math.sqrt(n_contributing)
        nn.init.normal_(self.r, mean=0, std=std)
        nn.init.uniform_(self.betas, 0, 2 * 3.14)


    @dimchecked
    def harmonics(self) -> ['f', 'd', 'd', 2]:
        real = torch.cos(self.order * self.angles + self.betas)
        imag = torch.sin(self.order * self.angles + self.betas)

        return complex(real, imag)

    @dimchecked
    def radial(self) -> ['f', 'd', 'd']:
        # pad radial function with 0 so that we can redirect out of bounds
        # accesses there
        r_ = torch.cat([
            self.r,
            torch.zeros(self.total_channels, 1, device=self.r.device)
        ], dim=1)

        rf = r_[:, self.floor]
        rc = r_[:, self.ceil]

        return self.alpha * rc + (1 - self.alpha) * rf


    def lowpass(self, w):
        return w #TODO: actual filtering

    @dimchecked
    def synthesize(self) -> ['f_out', 'f_in', 'r', 'r', 2]:
        radial = self.radial().unsqueeze(-1)
        harmonics = self.harmonics()

        kernel = self.lowpass(radial * harmonics)
        kernel = kernel.reshape(
            self.out_channels, self.in_channels, self.diam, self.diam, 2
        )

        return kernel


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

            checker.check(stream, ['n', -1, 'hi', 'wi', 2], name='in_stream {}'.format(i))

        out_streams = [(0 if repr != 0 else None) for repr in self.out_repr]

        for in_ord, in_stream in enumerate(streams):
            if stream is None:
                continue

            for out_ord in range(len(out_streams)):
                if out_streams[out_ord] is None:
                    continue

                conv = self.convs[ords2s(in_ord, out_ord)]
                out_streams[out_ord] += conv(in_stream)

        for i, stream in enumerate(out_streams):
            if stream is None:
                continue

            checker.check(stream, ['n', -1, 'ho', 'wo', 2], name='out_stream {}'.format(i))

        return out_streams
