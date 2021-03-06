from torch_dimcheck import dimchecked

from .._normalization import _Normalization

class BatchNorm2d(_Normalization):
    def __init__(self, repr, eps=1e-2):
        super(BatchNorm2d, self).__init__(repr, eps=eps, dim=2, kind='batch')

    @dimchecked
    def forward(self, x: [2, 'b', 'f', 'w', 'h']) -> [2, 'b', 'f', 'w', 'h']:
        return super(BatchNorm2d, self).forward(x)
