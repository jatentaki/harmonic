from torch_dimcheck import dimchecked

from .._convolution import _HConv, cconv_nd

@dimchecked
def complex_conv(x: [2, 'b',     'f_in', 'hx', 'wx', 'dx'],
                 w: [2, 'f_out', 'f_in', 'hk', 'wk', 'dk'],
                 pad=False) -> [2, 'b', 'f_out', 'ho', 'wo', 'do']:
    return cconv_nd(x, w, pad=pad, dim=3)


class HConv3d(_HConv):
    def __init__(self, in_repr, out_repr, size, radius=None, pad=False):
        super(HConv3d, self).__init__(
            in_repr, out_repr, size, pad=pad, dim=3, radius=radius
        )

    @dimchecked
    def forward(self,
                x: [2, 'b', 'fi', 'hx', 'wx', 'dx']) -> [2, 'b', 'fo', 'ho', 'wo', 'do']:
        return super(HConv3d, self).forward(x)
