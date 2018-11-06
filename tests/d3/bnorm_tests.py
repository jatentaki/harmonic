import torch, unittest
from harmonic.d3.bnorm import BatchNorm3d
from utils import rot90

class BNormTests(unittest.TestCase):
    def test_equivariance_eval(self):
        b, h, w, d = 5, 30, 30, 30
        repr = (2, 3)
        bnorm = BatchNorm3d(repr).double()

        inp = torch.randn(b, sum(repr), h, w, d, 2, dtype=torch.float64)
        rot = rot90(inp)

        bnorm.eval()
        base_fwd = bnorm(inp)
        rot_fwd = bnorm(rot)

        diff = (rot90(base_fwd) - rot_fwd).max().item()
        self.assertLess(diff, 1e-5)

    def test_equivariance_train(self):
        b, h, w, d = 5, 30, 30, 30
        repr = (2, 3)
        bnorm = BatchNorm3d(repr).double()

        inp = torch.randn(b, sum(repr), h, w, d, 2, dtype=torch.float64)
        rot = rot90(inp)

        # train mode
        bnorm.train()
        base_fwd = bnorm(inp)
        rot_fwd = bnorm(rot)

        diff = (rot90(base_fwd) - rot_fwd).max().item()
        self.assertLess(diff, 1e-5)

unittest.main()
