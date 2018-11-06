from torch_dimcheck import dimchecked

from .._convolution import _HConv, cconv_nd

@dimchecked
def complex_conv(x: ['b',     'f_in', 'hx', 'wx', 'dx', 2],
                 w: ['f_out', 'f_in', 'hk', 'wk', 'dk', 2],
                 pad=False) -> ['b', 'f_out', 'ho', 'wo', 'do', 2]:
    return cconv_nd(x, w, pad=pad, dim=3)


class HConv3d(_HConv):
    def __init__(self, in_repr, out_repr, radius, pad=False):
        super(HConv3d, self).__init__(
            in_repr, out_repr, radius, pad=pad, dim=3
        )

    @dimchecked
    def forward(self,
                x: ['b', 'fi', 'hx', 'wx', 'dx', 2]) -> ['b', 'fo', 'ho', 'wo', 'do', 2]:
        return super(HConv3d, self).forward(x)
