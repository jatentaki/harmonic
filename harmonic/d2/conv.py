from torch_dimcheck import dimchecked

from .._convolution import _StreamHConv, _HConv, cconv_nd

@dimchecked
def complex_conv(x: ['b',     'f_in', 'hx', 'wx', 2],
                 w: ['f_out', 'f_in', 'hk', 'wk', 2],
                 pad=False) -> ['b', 'f_out', 'ho', 'wo', 2]:
    return cconv_nd(x, w, pad=pad, dim=2)

class StreamHConv2d(_StreamHConv):
    def __init__(self, in_channels, out_channels, size, order,
                 radius=None, pad=False, name=None):

        super(StreamHConv2d, self).__init__(
            in_channels, out_channels, size, order, radius=radius, pad=pad, dim=2, name=name
        )


    @dimchecked
    def forward(self, t: ['b', 'fi', 'hi', 'wi', 2]) -> ['b', 'fo', 'ho', 'wo', 2]:

        return super(StreamHConv2d, self).forward(t)


class HConv2d(_HConv):
    def __init__(self, in_repr, out_repr, radius, pad=False):
        super(HConv2d, self).__init__(
            in_repr, out_repr, radius, pad=pad, dim=2
        )
