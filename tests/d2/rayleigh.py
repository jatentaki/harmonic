import torch, unittest
from harmonic.d2 import RayleighNorm2d
from harmonic.cmplx import magnitude
from utils import rot90

class RNormTests(unittest.TestCase):
    def test_featurewise_normalization(self):
        b, h, w = 5, 30, 40

        repr = (2, 3)
        rnorm = RayleighNorm2d(repr, eps=0.)

        inp = torch.randn(2, b, sum(repr), h, w)
        feature_means = torch.randn(sum(repr)).reshape(1, 1, -1, 1, 1)
        batch_means = torch.randn(b).reshape(1, -1, 1, 1, 1)
        feature_stds = torch.randn(sum(repr)).reshape(1, 1, -1, 1, 1)
        batch_stds = torch.randn(b).reshape(1, -1, 1, 1, 1)
        inp *= feature_stds
        inp *= batch_stds
        inp += feature_means
        inp += batch_means

        out = rnorm(inp)

        for batch_item in range(b):
            for feature in range(sum(repr)):
                slice_ = out[:, batch_item, feature, ...]
                self.assertLess(slice_.std() - 1., 0.01)

    def test_equivariance(self):
        b, h, w = 5, 30, 30
        repr = (2, 3)
        rnorm = RayleighNorm2d(repr).double()

        inp = torch.randn(2, b, sum(repr), h, w, dtype=torch.float64)
        rot = rot90(inp)

        # train mode
        rnorm.train()
        base_fwd = rnorm(inp)
        rot_fwd = rnorm(rot)

        diff = (rot90(base_fwd) - rot_fwd).max().item()
        self.assertLess(diff, 1e-5)

    def test_backward_jit(self):
        b, h, w = 5, 30, 30
        repr = (2, 3)
        rnorm = RayleighNorm2d(repr).double()

        inp = torch.randn(2, b, sum(repr), h, w, dtype=torch.float64)
        rot = rot90(inp)

        base_fwd = rnorm(inp)
        rot_fwd = rnorm(rot)

        diff = (rot90(base_fwd) - rot_fwd).max().item()
        self.assertLess(diff, 1e-5)
    

unittest.main()
