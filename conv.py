import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
# tensor_fmt = [batch, order, feature, height, width]
# weight_fmt = [in.order, in.feature, in.height, in.width, out.order, out.feature]

def _typechk(obj, type_):
    if not isinstance(obj, type_):
        fmt = "Expected {}, got {}"
        msg = fmt.format(type_, type(obj))
        raise ValueError(msg)

def _tenchk(t, shape=None, dtype=None):
    _typechk(t, torch.Tensor)

    def shape_matches(actual, expected):
        if len(actual) != len(expected):
            return False
        for a_s, e_s in zip(actual, expected):
            if e_s != -1 and e_s != a_s:
                return False

        return True

    if shape is not None and not shape_matches(t.shape, shape):
        fmt = "Expected shape {}, got {}"
        msg = fmt.format(shape, t.shape)
        raise ValueError(msg)

    if dtype is not None and t.dtype != dtype:
        fmt = "Expected dtype {}, got {}"
        msg = fmt.format(dtype, t.dtype)
        raise ValueError(msg)

    
class CTen:
    def __init__(self, cten):
        _tenchk(cten)
        if cten.shape[-1] != 2:
            fmt = "Can only construct CTen out of tensor which has list dim of 2. Got {}"
            msg = fmt.format(cten.shape)
            raise ValueError(msg)

        self.t = cten


    @staticmethod
    def from_parts(real, imag):
        _typechk(real, torch.Tensor)
        _typechk(imag, torch.Tensor)

        if not real.shape == imag.shape:
            fmt = "Size mismatch between real and imag ({} vs {})"
            msg = fmt.format(real.shape, imag.shape)
            raise ValueError(msg)

        t = torch.stack([real, imag], dim=len(real.shape))

        return CTen(t)


    @property
    def real(self):
        return self.t[..., 0]
    
    @property
    def imag(self):
        return self.t[..., 1]

    
def h_conv(x, w, pad=False):
    _typechk(x, CTen)
    _typechk(w, CTen)

    if pad:
        padding = w.shape[3] // 2
    else:
        padding = 0

    real = F.conv2d(x.real, w.real, padding=padding) - \
           F.conv2d(x.imag, w.imag, padding=padding)

    imag = F.conv2d(x.real, w.imag, padding=padding) + \
           F.conv2d(x.imag, w.real, padding=padding)

    return CTen.from_parts(real, imag)

class Weights:
    def __init__(self, r, phi, order):
        '''
            r - [n_features, radius]
            phi - [n_features]
        '''
        _tenchk(r, dtype=torch.float32)

        self.n_features = r.shape[0]
        self.radius = r.shape[1] - 1

        _tenchk(phi, shape=(self.n_features,), dtype=torch.float32)

        self.r = r
        self.diam = 2 * self.radius + 1

        self.phi = phi.reshape(-1, 1, 1)
        self.order = order

        self.precompute_bilinear()
        self.precompute_angles()
    
    def precompute_bilinear(self):
        # pad radial function with 0 so that we can redirect out of bounds
        # accesses there
        self.r_ = torch.cat([self.r, torch.zeros(self.n_features, 1)], dim=1)

        # compute radii on grid
        xs = torch.linspace(-self.radius, self.radius, self.diam)
        xs = xs.reshape(1, -1)
        ys = xs.reshape(-1, 1)
        rs = torch.sqrt(xs ** 2 + ys ** 2)

        # compute floor, ceil and alpha for bilinear interpolation
        floor = torch.floor(rs)
        self.alpha = (rs - floor).reshape(1, self.diam, self.diam)
        self.floor = floor.to(torch.int64)
        self.ceil = torch.ceil(rs).to(torch.int64)

        # redirect out of bounds accesses to the 0 we introduced earlier
        self.ceil[self.ceil > self.radius] = self.radius + 1
        self.floor[self.ceil > self.radius] = self.radius + 1


    def precompute_angles(self):
        # compute angles on grid
        xs = torch.linspace(-1, 1, self.diam)
        xs = xs.reshape(1, -1)
        ys = xs.reshape(-1, 1)
        self.angles = torch.atan2(xs, ys).unsqueeze(0)

    def harmonics(self):
        real = torch.cos(self.order * self.angles + self.phi)
        imag = torch.sin(self.order * self.angles + self.phi)

        return CTen.from_parts(real, imag) 


    def radial(self):
        rf = self.r_[:, self.floor]
        rc = self.r_[:, self.ceil]
        return self.alpha * rc + (1 - self.alpha) * rf


    def lowpass(self, w):
        return w #TODO: actual filtering

    def synthesize(self):
        radial = self.radial().unsqueeze(-1)
        harmonics = self.harmonics()

        print(radial.shape, harmonics.t.shape)
        return self.lowpass(CTen(radial * harmonics.t))
