from torch_dimcheck import dimchecked

from .._normalization import _Normalization

class InstanceNorm2d(_Normalization):
    def __init__(self, repr, eps=1e-2):
        super(InstanceNorm2d, self).__init__(repr, eps=eps, dim=2, kind='instance')

    @dimchecked
    def forward(self, x: [2, 'b', 'f', 'w', 'h']) -> [2, 'b', 'f', 'w', 'h']:
        return super(InstanceNorm2d, self).forward(x)
