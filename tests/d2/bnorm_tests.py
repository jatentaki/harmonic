import torch, unittest
from harmonic.d2 import BatchNorm2d
from harmonic.cmplx import magnitude
from utils import rot90

class BNormTests(unittest.TestCase):
    def test_featurewise_normalization(self):
        b, h, w = 5, 30, 30

        repr = (2, 3)
        bnorm = BatchNorm2d(repr)

        inp = torch.randn(2, b, sum(repr), h, w)
        feature_means = torch.randn(5).reshape(1, 1, -1, 1, 1)
        batch_means = torch.randn(b).reshape(1, -1, 1, 1, 1)
        feature_stds = torch.randn(5).reshape(1, 1, -1, 1, 1)
        batch_stds = torch.randn(b).reshape(1, -1, 1, 1, 1)
        inp *= feature_stds
        inp *= batch_stds
        inp += feature_means
        inp += batch_means
        
        out = bnorm(inp)
        mags = magnitude(out)

        for feature in range(sum(repr)):
            mag = mags[:, :, feature, ...]
            val = out[:, :, feature, ...]

            diff_std = torch.abs(mag.std() - 1.)
            diff_mean = torch.abs(val.mean())
            self.assertLess(diff_mean.item(), 0.1)
            self.assertLess(diff_std.item(), 0.1)

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
