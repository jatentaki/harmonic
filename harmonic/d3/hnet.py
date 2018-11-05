import torch
import torch.nn as nn

from torch_localize import localized_module
from torch_dimcheck import dimchecked

from .conv import CrossConv
from .nonl import ScalarGate
from .pooling import avg_pool2d
from .bnorm import MultiBNorm


hnet_default_layout = [
    (1, ),
    (5, 5, 5),
    (3, 2),
    (1, ),
]

@localized_module
class HNetBlock(nn.Module):
    def __init__(self, in_repr, out_repr, radius, first_nonl=True, pad=False):
        super(HNetBlock, self).__init__()

        self.in_repr = in_repr
        self.out_repr = out_repr
        self.first_nonl = first_nonl

        if first_nonl:
            self.bnorm = MultiBNorm(self.in_repr)
            self.nonl = ScalarGate(in_repr)
        self.conv = CrossConv(in_repr, out_repr, radius, pad=pad)

    def forward(self, *x):
        y = x
        if self.first_nonl:
            y = self.bnorm(*y)
            y = self.nonl(*y)
        y = self.conv(*y)

        return y


class HNet(nn.Module):
    def __init__(self, radius, layout=hnet_default_layout, pad=False):
        super(HNet, self).__init__()
        
        self.seq = nn.ModuleList()
        for i, (prev, next) in enumerate(zip(layout[:-1], layout[1:])):
            first_nonl = i != 0
            block = HNetBlock(
                prev, next, radius, pad=pad, first_nonl=first_nonl,
                name='hblock{}'.format(i)
            )
            self.seq.append(block)
        

    @dimchecked
    def forward(self, x: ['n', 1, 'wi', 'hi']) -> ['n', -1, 'wo', 'ho', 2]:
        x_cmplx = torch.stack([x, torch.zeros_like(x)], dim=-1)
        
        y_cmplx = (x_cmplx, )
        for block in self.seq:
            y_cmplx = block(*y_cmplx)

        return y_cmplx[0]
