from torch_dimcheck import dimchecked

from .._convolution import _HConv, cconv_nd

@dimchecked
def complex_conv(x: [2, 'b',     'f_in', 'hx', 'wx', 'dx'],
                 w: [2, 'f_out', 'f_in', 'hk', 'wk', 'dk'],
                 pad=False, transpose=False) -> [2, 'b', 'f_out', 'ho', 'wo', 'do']:
    return cconv_nd(x, w, pad=pad, dim=3, transpose=transpose)


class HConv3d(_HConv):
    def __init__(self, in_repr, out_repr, radius, pad=False):
        super(HConv3d, self).__init__(
            in_repr, out_repr, radius, pad=pad, dim=3, transpose=False
        )

    @dimchecked
    def forward(self, x: [2, 'b', 'fi', 'hx', 'wx', 'dx']
               )      -> [2, 'b', 'fo', 'ho', 'wo', 'do']:
        return super(HConv3d, self).forward(x)

class HConvTranspose3d(_HConv):
    def __init__(self, in_repr, out_repr, radius, pad=False):
        super(HConvTranspose3d, self).__init__(
            in_repr, out_repr, radius, pad=pad, dim=3, transpose=True
        )

    @dimchecked
    def forward(self, x: [2, 'b', 'fi', 'hx', 'wx', 'dx']
               )      -> [2, 'b', 'fo', 'ho', 'wo', 'do']:
        return super(HConvTranspose3d, self).forward(x)
