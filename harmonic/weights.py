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

        # for filters of order different than 0, the total contribution of element
        # r=0 is 0, so we skip it
        self.radial = nn.Parameter(
            torch.randn(self.total_channels, self.n_rings - 1, requires_grad=True)
        )
        self.angular = nn.Parameter(
            torch.randn(self.total_channels, self.n_rings - 1, requires_grad=True)
        )

        self.precompute_grid()
        self.register_buffer('gauss_interp', self.precompute_gaussian())

        if self.order == 0:
            self.center = nn.Parameter(
                torch.randn(self.total_channels, requires_grad=True)
            )
            gauss_center = self.precompute_center().reshape(1, 1, self.size, self.size)
            self.register_buffer('gauss_center', gauss_center)

        if initialize:
            self.initialize_weights()

    
    def precompute_grid(self):
        # note: we need to sample angles in [0, 2*pi) thus we take 1 more
        # sample than necessary and discard it
        angular_grid = torch.linspace(0, 2 * math.pi, self.n_angles + 1)[:-1]
        # note: we drop the first radial item since it's special-cased
        radial_grid = torch.linspace(0, self.radius, self.n_rings)[1:]
    
        self.register_buffer('angular_grid', angular_grid)
        self.register_buffer('radial_grid', radial_grid)
    

    @dimchecked
    def precompute_gaussian(self, sigma=0.5) -> ['d', 'd', 'nw', 'na']:
        '''
            Build a [self.size, self.size, self.n_rings - 1, self.n_angles] matrix
            which interpolates kernel coordinates from radial to Cartesian
            by means of Gaussian interpolation. The r=0 element is omitted
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

        # evaluate Gaussian function on distances
        gauss_interp = torch.exp(- dist2 / (2 * sigma ** 2))
        gauss_interp /= gauss_interp.sum(dim=(2, 3), keepdim=True)

        return gauss_interp

    @dimchecked
    def precompute_center(self, sigma=0.5) -> ['d', 'd']:
        # create coordinates
        xs = torch.linspace(-self.size / 2, self.size / 2, self.size).reshape(-1, 1)
        ys = xs.reshape(1, -1)
        
        # compute distance
        dist2 = xs.pow(2) + ys.pow(2)

        # evaluate Gaussian function on distances
        gauss_center = torch.exp(- dist2 / (2 * sigma ** 2))
        gauss_center /= gauss_center.sum()

        return gauss_center


    def initialize_weights(self):
        # we want to initialize such that assuming input from N(0, 1) the output
        # is as well in N(0, 1). This means each weight should also be from
        # Gaussian with mean 0 and sigma = 2 / sqrt(n_contributing_pixels).
        
        n_contributing = self.total_channels * self.size ** 2
        std = 2. / math.sqrt(n_contributing)
        nn.init.normal_(self.radial, mean=0, std=std)
        nn.init.uniform_(self.angular, 0, 2 * math.pi)

        if self.order == 0:
            nn.init.normal_(self.center, mean=0, std=std)


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
        r = self.n_rings - 1
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
        interpolated = torch.einsum('cfra,dera->cfde', (polar_harm, self.gauss_interp))

        if self.order == 0:
            interpolated += self.center.reshape(1, -1, 1, 1) * self.gauss_center

        return interpolated
