from torch_dimcheck import dimchecked

from .._batch_norm import _BatchNorm

class BatchNorm2d(_BatchNorm):
    def __init__(self, repr, eps=1e-2):
        super(BatchNorm2d, self).__init__(repr, eps=eps, dim=2)

    @dimchecked
    def forward(self, x: ['b', 'f', 'w', 'h', 2]) -> ['b', 'f', 'w', 'h', 2]:
        return super(BatchNorm2d, self).forward(x)
