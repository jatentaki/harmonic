import torch, itertools
import torch.nn as nn
from torch_localize import localized_module
from torch_dimcheck import ShapeChecker
from conv import HConv

'''
Glossary:
    stream - a set of hidden layers of same rotation order
'''

def ords2s(in_ord, out_ord):
    return '{}_{}'.format(in_ord, out_ord)

@localized_module
class CrossConv(nn.Module):
    def __init__(self, in_repr, out_repr, radius, pad=False):
        super(CrossConv, self).__init__()

        self.in_repr = in_repr
        self.out_repr = out_repr

        self.convs = nn.ModuleDict()

        # create an HConv which maps between all pairs on (input, output) streams
        for (in_ord, in_mult), (out_ord, out_mult) in itertools.product(
                                    enumerate(in_repr),
                                    enumerate(out_repr)):

            if in_mult == 0 or out_mult == 0:
                # either order is not represented in current (in, out) pair
                continue

            name = 'HConv {}x{} -> {}x{}'.format(in_mult, in_ord, out_mult, out_ord)
            conv = HConv(in_mult, out_mult, radius, in_ord - out_ord, pad=pad, name=name)
            self.convs[ords2s(in_ord, out_ord)] = conv

    def forward(self, *streams):
        if len(streams) != len(self.in_repr):
            fmt = "Based on repr {} expected {} streams, got {}"
            msg = fmt.format(self.in_repr, len(self.in_repr), len(streams))
            raise ValueError(msg)

        checker = ShapeChecker()
        for i, stream in enumerate(streams):
            checker.check(stream, ['n', -1, 'hi', 'wi', 2], name='in_stream {}'.format(i))

        out_streams = [(0 if repr != 0 else None) for repr in self.out_repr]

        for in_ord, in_stream in enumerate(streams):
            if stream is None:
                continue

            for out_ord in range(len(out_streams)):
                if out_streams[out_ord] is None:
                    continue

                conv = self.convs[ords2s(in_ord, out_ord)]
                out_streams[out_ord] += conv(in_stream)

        for i, stream in enumerate(out_streams):
            checker.check(stream, ['n', -1, 'ho', 'wo', 2], name='out_stream {}'.format(i))

        return out_streams
            

hnet_default_layout = [
    (1, ),
    (5, 5, 5),
    (3, 2),
    (1, ),
]

class HNet(nn.Module):
    def __init__(self, radius, layout=hnet_default_layout, pad=False):
        super(HNet, self).__init__()
        
        self.seq = nn.ModuleList()
        for i, (prev, next) in enumerate(zip(layout[:-1], layout[1:])):
            self.seq.append(CrossConv(prev, next, radius, pad=pad, name='hconv{}'.format(i)))

    def forward(self, x: ['n', 1, 'wi', 'hi']) -> ['n', 1, 'wo', 'ho']:
        x_cmplx = torch.stack([x, torch.zeros_like(x)], dim=-1)
        
        y_cmplx = (x_cmplx, )
        for hconv in self.seq:
            y_cmplx = hconv(*y_cmplx)

        return y_cmplx[0][..., 0]
