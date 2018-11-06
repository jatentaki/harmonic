from torch_dimcheck import dimchecked

from .._batch_norm import _BatchNorm

class BatchNorm3d(_BatchNorm):
    def __init__(self, repr, momentum=0.1, eps=1e-5):
        super(BatchNorm3d, self).__init__(repr, momentum=momentum, eps=eps, dim=3)

    def forward(self, x: ['b', 'f', 'w', 'h', 'd', 2]) -> ['b', 'f', 'w', 'h', 'd', 2]:
        return super(BatchNorm3d, self).forward(x)
