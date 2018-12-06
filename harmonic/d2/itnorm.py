from torch_dimcheck import dimchecked

from .._normalization import _Normalization

class ItemNorm2d(_Normalization):
    def __init__(self, repr, eps=1e-2):
        super(ItemNorm2d, self).__init__(repr, eps=eps, dim=2, kind='item')

    @dimchecked
    def forward(self, x: [2, 'b', 'f', 'w', 'h']) -> [2, 'b', 'f', 'w', 'h']:
        return super(ItemNorm2d, self).forward(x)
