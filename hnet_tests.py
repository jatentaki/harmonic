import torch, unittest
from hnet import CrossConv, HNet
from cmplx import magnitude
from utils import rot90

class HNetTests(unittest.TestCase):
    def test_instantiation(self):
        hnet = HNet(5)

    def test_equivariance(self):
        hnet = HNet(5)
        inp = torch.randn(7, 1, 60, 60)
        inp_rot = rot90(inp)

        out = hnet(inp)
        out_rot = hnet(inp_rot)

        diff = (magnitude(out) - magnitude(rot90(out_rot, k=3))).max().item()

        self.assertLess(diff, 1e-3)


unittest.main()
