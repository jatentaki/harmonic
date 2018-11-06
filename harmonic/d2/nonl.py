from .._scalar_gate import _ScalarGate

class ScalarGate(_ScalarGate):
    def __init__(self, repr):
        super(ScalarGate, self).__init__(repr, dim=2)

    def forward(self, x: ['b', 'f', 'h', 'w', 2]) -> ['b', 'f', 'h', 'w', 2]:
        return super(ScalarGate, self).forward(x)
