from torch_dimcheck import dimchecked

from .._convolution import _HConv, cconv_nd

@dimchecked
def complex_conv(x: [2, 'b',     'f_in', 'hx', 'wx', 'dx'],
                 w: [2, 'f_out', 'f_in', 'hk', 'wk', 'dk'],
                 transpose=False, **kwargs) -> [2, 'b', 'f_out', 'ho', 'wo', 'do']:
    return cconv_nd(x, w, dim=3, transpose=transpose, **kwargs)


class HConv3d(_HConv):
    def __init__(self, in_repr, out_repr, radius, **kwargs):
        super(HConv3d, self).__init__(
            in_repr, out_repr, radius, dim=3, transpose=False, conv_kwargs=kwargs
        )

    @dimchecked
    def forward(self, x: [2, 'b', 'fi', 'hx', 'wx', 'dx']
               )      -> [2, 'b', 'fo', 'ho', 'wo', 'do']:
        return super(HConv3d, self).forward(x)

class HConvTranspose3d(_HConv):
    def __init__(self, in_repr, out_repr, radius, **kwargs):
        super(HConvTranspose3d, self).__init__(
            in_repr, out_repr, radius, dim=3, transpose=True, conv_kwargs=kwargs
        )

    @dimchecked
    def forward(self, x: [2, 'b', 'fi', 'hx', 'wx', 'dx']
               )      -> [2, 'b', 'fo', 'ho', 'wo', 'do']:
        return super(HConvTranspose3d, self).forward(x)
