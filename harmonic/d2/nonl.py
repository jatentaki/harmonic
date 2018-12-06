from .._scalar_gate import _ScalarGate

class ScalarGate2d(_ScalarGate):
    def __init__(self, repr, mult=2):
        super(ScalarGate2d, self).__init__(repr, dim=2, mult=mult)

    def forward(self, x: [2, 'b', 'f', 'h', 'w']) -> [2, 'b', 'f', 'h', 'w']:
        return super(ScalarGate2d, self).forward(x)
