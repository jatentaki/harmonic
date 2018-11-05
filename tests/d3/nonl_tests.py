import torch, unittest
from harmonic.d3.nonl import ScalarGate

class ScalarGateTests(unittest.TestCase):
    def test_instantiation(self):
        nonl = ScalarGate((3, 6, 0, 1))

    def test_forward(self):
        nonl = ScalarGate((3, 6, 0, 1))
        
        n, h, w, d = 3, 40, 40

        inputs = [
            torch.randn(n, 3, h, w, d, 2),
            torch.randn(n, 6, h, w, d, 2),
            None,
            torch.randn(n, 1, h, w, d, 2)
        ]

        output = nonl(*inputs)


unittest.main()
