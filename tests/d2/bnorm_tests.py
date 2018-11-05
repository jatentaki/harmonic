import torch, unittest
from harmonic.d2.bnorm import StreamBatchNorm2d
from utils import rot90

class BNormTests(unittest.TestCase):
    def test_equivariance_eval(self):
        b, f, h, w = 5, 10, 30, 30
        bnorm = StreamBatchNorm2d(f).double()

        inp = torch.randn(b, f, h, w, 2, dtype=torch.float64)
        rot = rot90(inp)

        bnorm.eval()
        base_fwd = bnorm(inp)
        rot_fwd = bnorm(rot)

        diff = (rot90(base_fwd) - rot_fwd).max().item()
        self.assertLess(diff, 1e-5)

    def test_equivariance_train(self):
        b, f, h, w = 5, 10, 30, 30
        bnorm = StreamBatchNorm2d(f).double()

        inp = torch.randn(b, f, h, w, 2, dtype=torch.float64)
        rot = rot90(inp)

        # train mode
        bnorm.train()
        base_fwd = bnorm(inp)
        rot_fwd = bnorm(rot)

        diff = (rot90(base_fwd) - rot_fwd).max().item()
        self.assertLess(diff, 1e-5)

unittest.main()
