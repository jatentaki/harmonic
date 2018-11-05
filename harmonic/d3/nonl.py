from .._scalar_gate import _ScalarGate

class ScalarGate(_ScalarGate):
    def __init__(self, repr):
        super(ScalarGate, self).__init__(repr, dim=3)
