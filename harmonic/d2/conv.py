from torch_dimcheck import dimchecked

from .._convolution import _HConv, cconv_nd

@dimchecked
def complex_conv(x: [2, 'b',     'f_in', 'hx', 'wx'],
                 w: [2, 'f_out', 'f_in', 'hk', 'wk'],
                 transpose=False, **kwargs) -> [2, 'b', 'f_out', 'ho', 'wo']:
    return cconv_nd(x, w, dim=2, transpose=transpose, **kwargs)


class HConv2d(_HConv):
    def __init__(self, in_repr, out_repr, radius, **kwargs):
        super(HConv2d, self).__init__(
            in_repr, out_repr, radius, dim=2, transpose=False, conv_kwargs=kwargs
        )

    @dimchecked
    def forward(self, x: [2, 'b', 'fi', 'hx', 'wx']) -> [2, 'b', 'fo', 'ho', 'wo']:
        return super(HConv2d, self).forward(x)

class HConvTranspose2d(_HConv):
    def __init__(self, in_repr, out_repr, radius, **kwargs):
        super(HConvTranspose2d, self).__init__(
            in_repr, out_repr, radius, dim=2, transpose=True, conv_kwargs=kwargs
        )

    @dimchecked
    def forward(self, x: [2, 'b', 'fi', 'hx', 'wx']) -> [2, 'b', 'fo', 'ho', 'wo']:
        return super(HConvTranspose2d, self).forward(x)
