from torch_dimcheck import dimchecked

from .._convolution import _HConv, cconv_nd

@dimchecked
def complex_conv(x: [2, 'b',     'f_in', 'hx', 'wx'],
                 w: [2, 'f_out', 'f_in', 'hk', 'wk'],
                 pad=False, transpose=False) -> [2, 'b', 'f_out', 'ho', 'wo']:
    return cconv_nd(x, w, pad=pad, dim=2, transpose=transpose)


class HConv2d(_HConv):
    def __init__(self, in_repr, out_repr, radius, pad=False):
        super(HConv2d, self).__init__(
            in_repr, out_repr, radius, pad=pad, dim=2, transpose=False
        )

    @dimchecked
    def forward(self, x: [2, 'b', 'fi', 'hx', 'wx']) -> [2, 'b', 'fo', 'ho', 'wo']:
        return super(HConv2d, self).forward(x)

class HConvTranspose2d(_HConv):
    def __init__(self, in_repr, out_repr, radius, pad=False):
        super(HConvTranspose2d, self).__init__(
            in_repr, out_repr, radius, pad=pad, dim=2, transpose=True
        )

    @dimchecked
    def forward(self, x: [2, 'b', 'fi', 'hx', 'wx']) -> [2, 'b', 'fo', 'ho', 'wo']:
        return super(HConvTranspose2d, self).forward(x)
