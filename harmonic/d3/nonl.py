from .._scalar_gate import _ScalarGate

class ScalarGate(_ScalarGate):
    def __init__(self, repr):
        super(ScalarGate, self).__init__(repr, dim=3)

    def forward(self, x: ['b', 'f', 'h', 'w', 'd', 2]) -> ['b', 'f', 'h', 'w', 'd', 2]:
        return super(ScalarGate, self).forward(x)
