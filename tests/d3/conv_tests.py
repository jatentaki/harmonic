import torch, unittest
from harmonic.d3.conv import HConv, CrossConv
from utils import rot90
from harmonic.cmplx import magnitude

class HConvTests(unittest.TestCase):
    def _diff_rotation(self, order, plane):
        b, r, c1, c2, h, w, d = 3, 5, 3, 8, 20, 20, 20
        conv1 = HConv(c1, c2, r, order).double()
        conv2 = HConv(c2, c1, r, -order).double()

        rotation = lambda t: rot90(t, plane=plane)

        inp = torch.randn(b, c1, h, w, d, 2, dtype=torch.float64)
        rot = rotation(inp)

        base_fwd = conv2(conv1(inp))
        rot_fwd = conv2(conv1(rot))

        return (rotation(base_fwd) - rot_fwd).max().item()
        
    def test_equivariance_3_4(self):
        for order in range(-4, 4):
            diff = self._diff_rotation(order, plane=(3, 4))
            self.assertLess(diff, 1e-5)

    def test_nonequivariance_2_4(self):
        for order in range(-4, 4):
            diff = self._diff_rotation(order, plane=(2, 4))
            self.assertGreater(diff, 1)

    def test_nonequivariance_2_3(self):
        for order in range(-4, 4):
            diff = self._diff_rotation(order, plane=(2, 3))
            self.assertGreater(diff, 1)


class CrossConvTests(unittest.TestCase):
    def test_equivariance_single_stream(self):
        b, s, h, w, d = 3, 5, 20, 20, 20

        rep1 = (2, )
        rep2 = (0, 0, 3)

        cconv1 = CrossConv(rep1, rep2, s).double()
        cconv2 = CrossConv(rep2, rep1, s).double()

        inp = torch.randn(b, rep1[0], h, w, d, 2, dtype=torch.float64)
        rot = rot90(inp)

        base_fwd = cconv2(*cconv1(inp))
        rot_fwd = cconv2(*cconv1(rot))

        # single output stream
        self.assertEqual(len(base_fwd), 1)
        self.assertEqual(len(rot_fwd), 1)

        diff = (rot90(base_fwd[0]) -  rot_fwd[0]).max().item()
        
        self.assertLess(diff, 1e-3)


    def test_equivariance_multi_stream(self):
        b, r, h, w, d = 3, 5, 20, 20, 20

        rep1 = (2, )
        rep2 = (1, 2, 3)

        cconv1 = CrossConv(rep1, rep2, r).double()
        cconv2 = CrossConv(rep2, rep1, r).double()

        inp = torch.randn(b, rep1[0], h, w, d, 2, dtype=torch.float64)
        rot = rot90(inp)

        base_fwd = cconv2(*cconv1(inp))
        rot_fwd = cconv2(*cconv1(rot))

        # single output stream
        self.assertEqual(len(base_fwd), 1)
        self.assertEqual(len(rot_fwd), 1)

        diff = (rot90(base_fwd[0]) - rot_fwd[0]).max().item()
        
        self.assertLess(diff, 1e-3)


    def test_equivariance_multi_stream_two_hops(self):
        b, r, h, w, d = 3, 5, 20, 20, 20

        rep1 = (2, )
        rep2 = (1, 2, 3)
        rep3 = (4, 5, 6)
        rep4 = (2, )

        cconv1 = CrossConv(rep1, rep2, r).double()
        cconv2 = CrossConv(rep2, rep3, r).double()
        cconv3 = CrossConv(rep3, rep4, r).double()

        inp = torch.randn(b, rep1[0], h, w, d, 2, dtype=torch.float64)
        rot = rot90(inp)

        base_fwd = cconv3(*cconv2(*cconv1(inp)))
        rot_fwd = cconv3(*cconv2(*cconv1(rot)))

        # single output stream
        self.assertEqual(len(base_fwd), 1)
        self.assertEqual(len(rot_fwd), 1)

        diff = (rot90(base_fwd[0]) - rot_fwd[0]).max().item()
        
        self.assertLess(diff, 1e-3)

unittest.main()
