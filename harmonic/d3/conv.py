from torch_dimcheck import dimchecked

from .._convolution import _HConv

class HConv3d(_HConv):
    def __init__(self, in_repr, out_repr, size, radius=None, conv_kwargs=dict()):
        super(HConv3d, self).__init__(
            in_repr, out_repr, size, dim=3, radius=radius,
            transpose=False, conv_kwargs=conv_kwargs
        )

    @dimchecked
    def forward(self,
                x: [2, 'b', 'fi', 'hx', 'wx', 'dx']) -> [2, 'b', 'fo', 'ho', 'wo', 'do']:
        return super(HConv3d, self).forward(x)

class HConv3dTranspose(_HConv):
    def __init__(self, in_repr, out_repr, size, radius=None, conv_kwargs=dict()):
        super(HConv3d, self).__init__(
            in_repr, out_repr, size, dim=3, radius=radius,
            transpose=True, conv_kwargs=conv_kwargs
        )

    @dimchecked
    def forward(self,
                x: [2, 'b', 'fi', 'hx', 'wx', 'dx']) -> [2, 'b', 'fo', 'ho', 'wo', 'do']:
        return super(HConv3d, self).forward(x)
