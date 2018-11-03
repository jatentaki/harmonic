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
    def __init__(self, in_channels, out_channels, size, order,
                 radius=None, pad=False):
        super(HConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size
        self.order = order

        self.radius = radius if radius is not None else size / 2 - 2
        self.pad = pad

        self.weights = Weights(in_channels, out_channels, size, self.radius, order)

    @dimchecked
    def forward(self, t: ['b', 'fi', 'hi', 'wi', 2]) -> ['b', 'fo', 'ho', 'wo', 2]:
        kernel = self.weights.synthesize()

        return h_conv(t, kernel, pad=self.pad)


class Weights(nn.Module):
    def __init__(self, in_channels, out_channels, size, radius, order,
                 initialize=True):

        super(Weights, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size
        self.radius = radius
        self.order = order

        self.total_channels = in_channels * out_channels
        self.n_weights = int(math.ceil(self.radius)) + 2
        self.middle = size / 2

        self.r = nn.Parameter(
            torch.randn(self.total_channels, self.n_weights, requires_grad=True)
        )
        self.betas = nn.Parameter(
            torch.randn(self.total_channels, self.n_weights, requires_grad=True)
        )

        self.precompute_gaussian()
        self.precompute_angles()
        if initialize:
            self.initialize_weights()

    
    def precompute_gaussian(self, sigma=0.5):
        '''
            Build a [self.diam, self.diam, self.n_weights] matrix which interpolates
            the radial function onto the square convolution kernel by means
            of Gaussian interpolation
        '''

        # compute radii on grid
        xs = torch.arange(self.size, dtype=torch.float32) - self.middle + 0.5
        xs = xs.reshape(1, -1)
        ys = xs.reshape(-1, 1)
        rs = torch.sqrt(xs ** 2 + ys ** 2)

        # compute a radial distance matrix between each grid point
        # and each radial function element
        weight_positions = torch.linspace(
            0, self.radius, self.n_weights,
            dtype=self.r.dtype, device=self.r.device
        )
        dist = rs.unsqueeze(2) - weight_positions.reshape(1, 1, -1)

        # evaluate Gaussian function on distances
        gauss = torch.exp(- dist ** 2 / (2 * sigma ** 2))
        gauss = gauss / gauss.sum(dim=(0, 1), keepdim=True)

        # self-assign the interpolation weights
        self.register_buffer('gauss', gauss)


    def precompute_angles(self):
        # compute angles on grid
        xs = torch.linspace(-1, 1, self.size)
        xs = xs.reshape(1, -1)
        ys = xs.reshape(-1, 1)
        self.register_buffer('angles', torch.atan2(xs, ys).unsqueeze(0))
    
    
    def initialize_weights(self):
        # we want to initialize such that assuming input from N(0, 1) the output
        # is as well in N(0, 1). This means each weight should also be from
        # Gaussian with mean 0 and sigma = 2 / sqrt(n_contributing_pixels).
        
        n_contributing = self.total_channels * self.size ** 2
        std = 2. / math.sqrt(n_contributing)
        nn.init.normal_(self.r, mean=0, std=std)
        nn.init.uniform_(self.betas, 0, 2 * 3.14)


    @dimchecked
    def gaussian_interp(self, f: ['f', 'r']) -> ['f', 'd', 'd']:
        '''
            Interpolate a radial function `f` onto 2d kernels by Gaussian interpolation
            (interpolation matrix precomputed in `precompute_gaussian`
        '''
        # d == e, einsum syntax doesn't allow expressing that
        return torch.einsum('fr,der->fde', (f, self.gauss))


    @dimchecked
    def harmonics(self) -> ['f', 'd', 'd', 2]:
        betas = self.gaussian_interp(self.betas)

        real = torch.cos(self.order * self.angles + betas)
        imag = torch.sin(self.order * self.angles + betas)

        return complex(real, imag)


    @dimchecked
    def radial(self) -> ['f', 'd', 'd']:
        return self.gaussian_interp(self.r)


    @dimchecked
    def synthesize(self) -> ['f_out', 'f_in', 'r', 'r', 2]:
#        import matplotlib.pyplot as plt
#        for i in range(self.gauss.shape[2]):
#            plt.imshow(self.gauss.numpy()[..., i])
#            plt.show()
        radial = self.radial().unsqueeze(-1)
        harmonics = self.harmonics()

        kernel = radial * harmonics
        kernel = kernel.reshape(
            self.out_channels, self.in_channels, self.size, self.size, 2
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
