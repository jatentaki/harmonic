import torch, unittest
from harmonic.d3 import ScalarGate3d

class ScalarGateTests(unittest.TestCase):
    def test_forward(self):
        nonl = ScalarGate3d((3, 6, 0, 1))
        
        n, h, w, d = 3, 40, 40, 40

        inputs = torch.randn(2, n, 3 + 6 + 1, h, w, d)
        output = nonl(inputs)


unittest.main()
