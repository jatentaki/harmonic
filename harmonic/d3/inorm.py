from torch_dimcheck import dimchecked

from .._normalization import _Normalization

class InstanceNorm3d(_Normalization):
    def __init__(self, repr, eps=1e-2):
        super(InstanceNorm3d, self).__init__(repr, eps=eps, dim=3, kind='instance')

    def forward(self, x: [2, 'b', 'f', 'w', 'h', 'd']) -> [2, 'b', 'f', 'w', 'h', 'd']:
        return super(InstanceNorm3d, self).forward(x)
