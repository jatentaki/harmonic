import torch, itertools, math
import torch.nn as nn
import torch.nn.functional as F
from torch_localize import localized_module
from torch_dimcheck import dimchecked, ShapeChecker

from cmplx import cmplx 

@dimchecked
def complex_conv(x: ['b',     'f_in', 'xh', 'xw', 2],
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

        self.weights = Weights(in_channels, out_channels, size, self.radius, order)

    @dimchecked
    def forward(self, t: ['b', 'fi', 'hi', 'wi', 2]) -> ['b', 'fo', 'ho', 'wo', 2]:
        kernel = self.weights.synthesize()

        return complex_conv(t, kernel, pad=self.pad)


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
        self.n_rings = int(math.ceil(self.radius))
        self.n_angles = 12 # FIXME: figure out he right number

        self.r = nn.Parameter(
            torch.randn(self.total_channels, self.n_rings, requires_grad=True)
        )
        self.betas = nn.Parameter(
            torch.randn(self.total_channels, self.n_rings, requires_grad=True)
        )

        self.precompute_grid()
        gauss = self.precompute_gaussian()
        self.register_buffer('gauss', gauss)

        if initialize:
            self.initialize_weights()

    
    def precompute_grid(self):
        # note: we need to sample angles in [0, 2*pi) thus we take 1 more
        # sample than necessary and discard it
        angles = torch.linspace(0, 2 * math.pi, self.n_angles + 1)[:-1]
        radii = torch.linspace(0, self.size / 2, self.n_rings)
    
        self.register_buffer('angles', angles)
        self.register_buffer('radii', radii)
    

    @dimchecked
    def precompute_gaussian(self, sigma=0.5) -> ['d', 'd', 'nw', 'na']:
        '''
            Build a [self.size, self.size, self.n_rings, self.n_angles] matrix
            which interpolates kernel coordinates from radial to Cartesian
            by means of Gaussian interpolation
        '''

        # compute cartesian coordinates of each polar grid point
        p_xs = self.radii.unsqueeze(1) * torch.cos(self.angles).unsqueeze(0)
        p_ys = self.radii.unsqueeze(1) * torch.sin(self.angles).unsqueeze(0)

        # compute cartesian coordinates of each cartesian grid point
        c_xs = torch.linspace(-self.size/2, self.size/2, self.size)
        c_ys = c_xs

        # compute distances across each cartesian axis of shape
        # [self.size, self.n_rings, self.n_angles]
        x_dist = p_xs.unsqueeze(0) - c_xs.reshape(-1, 1, 1)
        y_dist = p_ys.unsqueeze(0) - c_ys.reshape(-1, 1, 1)

        # compute matrix of distances between gridpoints in both
        # coordinate systems of shape
        # [self.size, self.size, self.n_rings, self.n_angles]
        dist2 = x_dist.pow(2).unsqueeze(0) + y_dist.pow(2).unsqueeze(1)
        dist = torch.sqrt(dist2)

        # evaluate Gaussian function on distances
        gauss = torch.exp(- dist ** 2 / (2 * sigma ** 2))
        gauss = gauss / gauss.sum(dim=(0, 1), keepdim=True)

        return gauss 


    def initialize_weights(self):
        # we want to initialize such that assuming input from N(0, 1) the output
        # is as well in N(0, 1). This means each weight should also be from
        # Gaussian with mean 0 and sigma = 2 / sqrt(n_contributing_pixels).
        
        n_contributing = self.total_channels * self.size ** 2
        std = 2. / math.sqrt(n_contributing)
        nn.init.normal_(self.r, mean=0, std=std)
        nn.init.uniform_(self.betas, 0, 2 * math.pi)


    @dimchecked
    def polar_harmonics(self) -> ['f', 'ring', 'angle', 2]:
        '''
            Synthesize filters in polar coordinates. Grid is given by `angles` and
            `radii`, parameters by `betas` (phase offset) and `r` (radial profile)
        '''
        real = torch.cos(
            self.order * self.angles.reshape(1, 1, -1) + self.betas.unsqueeze(2)
        )
        imag = torch.sin(
            self.order * self.angles.reshape(1, 1, -1) + self.betas.unsqueeze(2)
        )

        real = real * self.r.unsqueeze(2)
        imag = imag * self.r.unsqueeze(2)

        return cmplx(real, imag)

    @dimchecked
    def cartesian_harmonics(self) -> ['f', 'd', 'd', 2]:
        '''
            Interpolate the results of `polar_harmonics()` onto Cartesian grid
        '''
        polar_harm = self.polar_harmonics()
        return torch.einsum('frac,dera->fdec', (polar_harm, self.gauss))


    @dimchecked
    def synthesize(self) -> ['f_out', 'f_in', 'r', 'r', 2]:

        kernel = self.cartesian_harmonics().reshape(
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

            checker.check(stream, ['n', -1, 'hi', 'wi', 2], name='in_stream{}'.format(i))

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

            checker.check(stream, ['n', -1, 'ho', 'wo', 2], name='out_stream{}'.format(i))

        return out_streams
