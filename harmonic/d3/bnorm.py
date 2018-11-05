from torch_dimcheck import dimchecked

from .._batch_norm import _BatchNorm, _StreamBatchNorm

class BatchNorm3d(_BatchNorm):
    def __init__(self, repr, momentum=0.1, eps=1e-5):
        super(BatchNorm3d, self).__init__(repr, momentum=momentum, eps=eps, dim=3)


class StreamBatchNorm3d(_StreamBatchNorm):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super(StreamBatchNorm3d, self).__init__(
            num_features, momentum=momentum, eps=eps, dim=3
        )

    @dimchecked
    def forward(self, x: ['b', 'f', 'w', 'h', 'd', 2]) -> ['b', 'f', 'w', 'h', 'd', 2]:
        return super(StreamBatchNorm3d, self).forward(x)
