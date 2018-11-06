import torch, unittest
from harmonic.d2.nonl import ScalarGate

class ScalarGateTests(unittest.TestCase):
    def test_forward(self):
        nonl = ScalarGate((3, 6, 0, 1))
        n, h, w = 3, 40, 40
        inputs = torch.randn(n, 3 + 6 + 1, h, w, 2)
        output = nonl(inputs)


unittest.main()
