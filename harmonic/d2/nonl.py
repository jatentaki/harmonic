from .._scalar_gate import _ScalarGate

class ScalarGate2d(_ScalarGate):
    def __init__(self, repr):
        super(ScalarGate2d, self).__init__(repr, dim=2)

    def forward(self, x: ['b', 'f', 'h', 'w', 2]) -> ['b', 'f', 'h', 'w', 2]:
        return super(ScalarGate2d, self).forward(x)
