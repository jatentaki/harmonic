import torch, unittest
from harmonic.d2 import GroupNorm2d
from harmonic.cmplx import magnitude
from utils import rot90

class GroupNormTests(unittest.TestCase):
    def test_equivariance_eval(self):
        b, h, w = 5, 30, 30

        repr = (2, 3)
        gnorm = GroupNorm2d(repr).double()

        inp = torch.randn(2, b, sum(repr), h, w, dtype=torch.float64)
        rot = rot90(inp)

        # eval mode
        gnorm.eval()
        base_fwd = gnorm(inp)
        rot_fwd = gnorm(rot)

        diff = (rot90(base_fwd) - rot_fwd).max().item()
        self.assertLess(diff, 1e-5)

    def test_equivariance_train(self):
        b, h, w = 5, 30, 30
        repr = (2, 3)
        gnorm = GroupNorm2d(repr).double()

        inp = torch.randn(2, b, sum(repr), h, w, dtype=torch.float64)
        rot = rot90(inp)

        # train mode
        gnorm.train()
        base_fwd = gnorm(inp)
        rot_fwd = gnorm(rot)

        diff = (rot90(base_fwd) - rot_fwd).max().item()
        self.assertLess(diff, 1e-5)

    def test_backward_jit(self):
        b, h, w = 5, 30, 30
        repr = (2, 3)
        gnorm = GroupNorm2d(repr).double()

        inp = torch.randn(2, b, sum(repr), h, w, dtype=torch.float64)
        rot = rot90(inp)

        # train mode
        gnorm.train()
        base_fwd = gnorm(inp)
        rot_fwd = gnorm(rot)

        diff = (rot90(base_fwd) - rot_fwd).max().item()
        self.assertLess(diff, 1e-5)

unittest.main()
