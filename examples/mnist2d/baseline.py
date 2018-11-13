import torch
import torch.nn as nn

from torch_localize import localized_module
from torch_dimcheck import dimchecked

from torch.nn import BatchNorm2d, Conv2d
from torch.nn.functional import avg_pool2d

hnet_default_layout = [
    (1, ),
    (5, 5, 5),
    (3, 2),
    (1, ),
]

class ScalarGate2d(nn.Module):
    def __init__(self, n_features):
        super(ScalarGate2d, self).__init__()

        self.conv1 = Conv2d(n_features, n_features, 1)
        self.conv2 = Conv2d(n_features, n_features, 1)

    def forward(self, x):
        g = self.conv1(x)
        g = nn.functional.relu(g)
        g = self.conv2(g)
        g = torch.sigmoid(g)

        return g * x


@localized_module
class BNetBlock(nn.Module):
    def __init__(self, in_features, out_features, size, first_nonl=True):
        super(BNetBlock, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.first_nonl = first_nonl

        if first_nonl:
            self.bnorm = BatchNorm2d(in_features)
            self.nonl = ScalarGate2d(in_features)
        self.conv = Conv2d(in_features, out_features, size)

    def forward(self, x):
        y = x
        if self.first_nonl:
            y = self.bnorm(y)
            y = self.nonl(y)
        y = self.conv(y)

        return y


class BNet(nn.Module):
    def __init__(self, size, layout=hnet_default_layout):
        super(BNet, self).__init__()
        
        layout_flat = []
        for layer in layout:
            if layer == (1, ):
                layout_flat.append(1)
            else:
                layout_flat.append(2 * sum(layer))

        seq = []
        for i, (prev, next) in enumerate(zip(layout_flat[:-1], layout_flat[1:])):
            first_nonl = i != 0
            block = BNetBlock(
                prev, next, size, first_nonl=first_nonl,
                name='hblock{}'.format(i)
            )
            seq.append(block)

        self.seq = nn.Sequential(*seq)
        

    @dimchecked
    def forward(self, x: ['n', 1, 'wi', 'hi']) -> ['n', 'fo', 'wo', 'ho']:
        return self.seq(x)
