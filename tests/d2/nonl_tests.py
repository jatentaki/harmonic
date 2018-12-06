import torch, unittest
from harmonic.d2 import ScalarGate2d

class ScalarGateTests(unittest.TestCase):
    def test_forward_m1(self):
        nonl = ScalarGate2d((3, 6, 0, 1), mult=1)
        n, h, w = 3, 40, 40
        inputs = torch.randn(2, n, 3 + 6 + 1, h, w)
        output = nonl(inputs)

    def test_forward_m2(self):
        nonl = ScalarGate2d((3, 6, 0, 1), mult=2)
        n, h, w = 3, 40, 40
        inputs = torch.randn(2, n, 3 + 6 + 1, h, w)
        output = nonl(inputs)

unittest.main()
