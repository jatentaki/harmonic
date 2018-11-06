import torch, unittest
from harmonic.d2.conv import HConv2d
from utils import rot90
from harmonic.cmplx import magnitude

class HConvTests(unittest.TestCase):
    def test_equivariance_0(self):
        self._test_equivariance(0)

    def test_equivariance_1(self):
        self._test_equivariance(1)

    def test_equivariance_2(self):
        self._test_equivariance(2)

    def test_equivariance_3(self):
        self._test_equivariance(3)

    def test_equivariance_4(self):
        self._test_equivariance(4)

    def test_equivariance_m1(self):
        self._test_equivariance(-1)

    def test_equivariance_m2(self):
        self._test_equivariance(-2)

    def test_equivariance_m3(self):
        self._test_equivariance(-3)

    def test_equivariance_m4(self):
        self._test_equivariance(-4)

    def _test_equivariance(self, order):
        b, s, c1, c2, h, w = 5, 7, 5, 10, 30, 30
        repr1 = [c1]
        repr2 = [0] * (order - 1) + [c2]

        conv1 = HConv2d(repr1, repr2, s).double()
        conv2 = HConv2d(repr2, repr1, s).double()

        inp = torch.randn(b, c1, h, w, 2, dtype=torch.float64)
        rot = rot90(inp)

        base_fwd = conv2(conv1(inp))
        rot_fwd = conv2(conv1(rot))

        diff = (rot90(base_fwd) - rot_fwd).max().item()
        
        self.assertLess(diff, 1e-5)

    def test_equivariance_single_stream(self):
        b, s, h, w = 5, 7, 50, 50

        rep1 = (2, )
        rep2 = (0, 0, 3)

        cconv1 = HConv2d(rep1, rep2, s).double()
        cconv2 = HConv2d(rep2, rep1, s).double()

        inp = torch.randn(b, rep1[0], h, w, 2, dtype=torch.float64)
        rot = rot90(inp)

        base_fwd = cconv2(cconv1(inp))
        rot_fwd = cconv2(cconv1(rot))

        diff = (rot90(base_fwd) -  rot_fwd).max().item()
        
        self.assertLess(diff, 1e-3)


    def test_equivariance_multi_stream(self):
        b, r, h, w = 5, 7, 50, 50

        rep1 = (2, )
        rep2 = (1, 2, 3)

        cconv1 = HConv2d(rep1, rep2, r).double()
        cconv2 = HConv2d(rep2, rep1, r).double()

        inp = torch.randn(b, rep1[0], h, w, 2, dtype=torch.float64)
        rot = rot90(inp)

        base_fwd = cconv2(cconv1(inp))
        rot_fwd = cconv2(cconv1(rot))

        diff = (rot90(base_fwd) - rot_fwd).max().item()
        
        self.assertLess(diff, 1e-3)


    def test_equivariance_multi_stream_two_hops(self):
        b, r, h, w = 5, 7, 50, 50

        rep1 = (2, )
        rep2 = (1, 2, 3)
        rep3 = (4, 5, 6)
        rep4 = (2, )

        cconv1 = HConv2d(rep1, rep2, r).double()
        cconv2 = HConv2d(rep2, rep3, r).double()
        cconv3 = HConv2d(rep3, rep4, r).double()

        inp = torch.randn(b, rep1[0], h, w, 2, dtype=torch.float64)
        rot = rot90(inp)

        base_fwd = cconv3(cconv2(cconv1(inp)))
        rot_fwd = cconv3(cconv2(cconv1(rot)))

        diff = (rot90(base_fwd) - rot_fwd).max().item()
        
        self.assertLess(diff, 1e-3)

unittest.main(failfast=True)
