from torch_dimcheck import dimchecked

from .._convolution import _HConv, cconv_nd

@dimchecked
def complex_conv(x: ['b',     'f_in', 'hx', 'wx', 2],
                 w: ['f_out', 'f_in', 'hk', 'wk', 2],
                 pad=False) -> ['b', 'f_out', 'ho', 'wo', 2]:
    return cconv_nd(x, w, pad=pad, dim=2)


class HConv2d(_HConv):
    def __init__(self, in_repr, out_repr, radius, pad=False):
        super(HConv2d, self).__init__(
            in_repr, out_repr, radius, pad=pad, dim=2
        )

    @dimchecked
    def forward(self, x: ['b', 'fi', 'hx', 'wx', 2]) -> ['b', 'fo', 'ho', 'wo', 2]:
        return super(HConv2d, self).forward(x)
