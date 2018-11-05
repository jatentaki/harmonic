from torch_dimcheck import dimchecked

from .._convolution import _HConv, _StreamHConv, cconv_nd

@dimchecked
def complex_conv(x: ['b',     'f_in', 'hx', 'wx', 'dx', 2],
                 w: ['f_out', 'f_in', 'hk', 'wk', 'dk', 2],
                 pad=False) -> ['b', 'f_out', 'ho', 'wo', 'do', 2]:
    return cconv_nd(x, w, pad=pad, dim=3)

class StreamHConv3d(_StreamHConv):
    def __init__(self, in_channels, out_channels, size, order,
                 radius=None, pad=False, name=None):

        super(StreamHConv3d, self).__init__(
            in_channels, out_channels, size, order, radius=radius, pad=pad, dim=3, name=name
        )


    @dimchecked
    def forward(self, t: ['b', 'fi', 'hi', 'wi', 'di', 2]) -> ['b', 'fo', 'ho', 'wo', 'do', 2]:

        return super(StreamHConv3d, self).forward(t)


class HConv3d(_HConv):
    def __init__(self, in_repr, out_repr, radius, pad=False):
        super(HConv3d, self).__init__(
            in_repr, out_repr, radius, pad=pad, dim=3
        )
