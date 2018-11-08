from .._scalar_gate import _ScalarGate

class ScalarGate3d(_ScalarGate):
    def __init__(self, repr):
        super(ScalarGate3d, self).__init__(repr, dim=3)

    def forward(self, x: [2, 'b', 'f', 'h', 'w', 'd']) -> [2, 'b', 'f', 'h', 'w', 'd']:
        return super(ScalarGate3d, self).forward(x)
