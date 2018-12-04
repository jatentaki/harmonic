import torch, math
import torch.nn as nn

from .cmplx import cmplx

from torch_localize import localized_module
from torch_dimcheck import dimchecked

@localized_module
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
        self.n_rings = int(math.ceil(self.radius)) + 1
        self.n_angles = 4 * (size - 1)

        self.radial = nn.Parameter(
            torch.randn(self.total_channels, self.n_rings, requires_grad=True)
        )
        self.angular = nn.Parameter(
            torch.randn(self.total_channels, self.n_rings, requires_grad=True)
        )

        self.precompute_grid()
        gauss_interp = self.precompute_gaussian()
        self.register_buffer('gauss_interp', gauss_interp)

        if initialize:
            self.initialize_weights()

    
    def precompute_grid(self):
        # note: we need to sample angles in [0, 2*pi) thus we take 1 more
        # sample than necessary and discard it
        angular_grid = torch.linspace(0, 2 * math.pi, self.n_angles + 1)[:-1]
        radial_grid = torch.linspace(0, self.radius, self.n_rings)
    
        self.register_buffer('angular_grid', angular_grid)
        self.register_buffer('radial_grid', radial_grid)
    

    @dimchecked
    def precompute_gaussian(self, sigma=0.5) -> ['d', 'd', 'nw', 'na']:
        '''
            Build a [self.size, self.size, self.n_rings, self.n_angles] matrix
            which interpolates kernel coordinates from radial to Cartesian
            by means of Gaussian interpolation
        '''

        # compute cartesian coordinates of each polar grid point
        p_xs = self.radial_grid.unsqueeze(1) * \
               self.angular_grid.cos().unsqueeze(0)
        p_ys = self.radial_grid.unsqueeze(1) * \
               self.angular_grid.sin().unsqueeze(0)

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
        norm = 1 / math.sqrt(2 * math.pi * sigma ** 2)
        gauss = torch.exp(- dist ** 2 / (2 * sigma ** 2)) / norm

        return gauss 


    def initialize_weights(self):
        # we want to initialize such that assuming input from N(0, 1) the output
        # is as well in N(0, 1). This means each weight should also be from
        # Gaussian with mean 0 and sigma = 2 / sqrt(n_contributing_pixels).
        
        n_contributing = self.total_channels * self.size ** 2
        std = 2. / math.sqrt(n_contributing)
        nn.init.normal_(self.radial, mean=0, std=std)
        nn.init.uniform_(self.angular, 0, 2 * math.pi)


    @dimchecked
    def polar_harmonics(self) -> [2, 'f', 'ring', 'angle']:
        '''
            Synthesize filters in polar coordinates. Grid is given by `angles` and
            `radii`, parameters by `angular` (phase offset) and `r` (radial profile)
        '''

        # FIXME: until an upcoming PR by Adam Paszke jit fails with operations
        # which depend on broadcasting. Therefore we have to manually call
        # `expand` on all operands to allow jiting
        f = self.total_channels
        r = self.n_rings
        a = self.n_angles

        real = torch.cos(
            self.order * self.angular_grid.reshape(1, 1, -1).expand(f, r, a) +
            self.angular.unsqueeze(2).expand(f, r, a)
        )
        imag = torch.sin(
            self.order * self.angular_grid.reshape(1, 1, -1).expand(f, r, a) +
            self.angular.unsqueeze(2).expand(f, r, a)
        )

        real = real * self.radial.unsqueeze(2).expand(f, r, a)
        imag = imag * self.radial.unsqueeze(2).expand(f, r, a)

        return cmplx(real, imag)

    @dimchecked
    def cartesian_harmonics(self) -> [2, 'f', 'd', 'd']:
        '''
            Interpolate the results of `polar_harmonics()` onto Cartesian grid
        '''
        polar_harm = self.polar_harmonics()
        return torch.einsum('cfra,dera->cfde', (polar_harm, self.gauss_interp))
