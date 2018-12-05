from torch_dimcheck import dimchecked

from .._convolution import _HConv

class HConv2d(_HConv):
    def __init__(self, in_repr, out_repr, size, radius=None, conv_kwargs=dict()):
        super(HConv2d, self).__init__(
            in_repr, out_repr, size, dim=2, radius=radius, 
            transpose=False, conv_kwargs=conv_kwargs
        )

    @dimchecked
    def forward(self, x: [2, 'b', 'fi', 'hx', 'wx']) -> [2, 'b', 'fo', 'ho', 'wo']:
        return super(HConv2d, self).forward(x)

class HConv2dTranspose(_HConv):
    def __init__(self, in_repr, out_repr, size, radius=None, conv_kwargs=dict()):
        super(HConv2d, self).__init__(
            in_repr, out_repr, size, dim=2, radius=radius,
            transpose=True, conv_kwargs=conv_kwargs
        )

    @dimchecked
    def forward(self, x: [2, 'b', 'fi', 'hx', 'wx']) -> [2, 'b', 'fo', 'ho', 'wo']:
        return super(HConv2d, self).forward(x)
