import torch, unittest
from harmonic.d2 import BatchNorm2d
from utils import rot90

class BNormTests(unittest.TestCase):
    def test_equivariance_eval(self):
        b, h, w = 5, 30, 30

        repr = (2, 3)
        bnorm = BatchNorm2d(repr).double()

        inp = torch.randn(2, b, sum(repr), h, w, dtype=torch.float64)
        rot = rot90(inp)

        # eval mode
        bnorm.eval()
        base_fwd = bnorm(inp)
        rot_fwd = bnorm(rot)

        diff = (rot90(base_fwd) - rot_fwd).max().item()
        self.assertLess(diff, 1e-5)

    def test_equivariance_train(self):
        b, h, w = 5, 30, 30
        repr = (2, 3)
        bnorm = BatchNorm2d(repr).double()

        inp = torch.randn(2, b, sum(repr), h, w, dtype=torch.float64)
        rot = rot90(inp)

        # train mode
        bnorm.train()
        base_fwd = bnorm(inp)
        rot_fwd = bnorm(rot)

        diff = (rot90(base_fwd) - rot_fwd).max().item()
        self.assertLess(diff, 1e-5)

    def test_backward_jit(self):
        b, h, w = 5, 30, 30
        repr = (2, 3)
        bnorm = BatchNorm2d(repr).double()

        inp = torch.randn(2, b, sum(repr), h, w, dtype=torch.float64)
        rot = rot90(inp)

        # train mode
        bnorm.train()
        base_fwd = bnorm(inp)
        rot_fwd = bnorm(rot)

        diff = (rot90(base_fwd) - rot_fwd).max().item()
        self.assertLess(diff, 1e-5)

unittest.main()
