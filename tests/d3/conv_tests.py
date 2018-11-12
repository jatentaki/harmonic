import torch, unittest
from harmonic.d3.conv import HConv3d
from utils import rot90

class HConvTests(unittest.TestCase):
    def _diff_rotation(self, order, plane):
        b, s, c1, c2, h, w, d = 3, 5, 3, 8, 20, 20, 20
        repr1 = [c1]
        repr2 = [0] * (order - 1) + [c2]

        conv1 = HConv3d(repr1, repr2, s).double()
        conv2 = HConv3d(repr2, repr1, s).double()

        rotation = lambda t: rot90(t, plane=plane)

        inp = torch.randn(2, b, c1, h, w, d, dtype=torch.float64)
        rot = rotation(inp)

        base_fwd = conv2(conv1(inp))
        rot_fwd = conv2(conv1(rot))

        return (rotation(base_fwd) - rot_fwd).max().item()
        
    def test_equivariance_x_y(self):
        for order in range(-4, 4):
            diff = self._diff_rotation(order, plane=(4, 5))
            self.assertLess(diff, 1e-5)

    def test_nonequivariance_z_y(self):
        for order in range(-4, 4):
            diff = self._diff_rotation(order, plane=(3, 5))
            self.assertGreater(diff, 1e-3)

    def test_nonequivariance_z_x(self):
        for order in range(-4, 4):
            diff = self._diff_rotation(order, plane=(3, 4))
            self.assertGreater(diff, 1e-3)

    def test_equivariance_single_stream(self):
        b, s, h, w, d = 3, 5, 20, 20, 20

        rep1 = (2, )
        rep2 = (0, 0, 3)

        cconv1 = HConv3d(rep1, rep2, s).double()
        cconv2 = HConv3d(rep2, rep1, s).double()

        inp = torch.randn(2, b, rep1[0], h, w, d, dtype=torch.float64)
        rot = rot90(inp)

        base_fwd = cconv2(cconv1(inp))
        rot_fwd = cconv2(cconv1(rot))

        diff = (rot90(base_fwd) -  rot_fwd).max().item()
        
        self.assertLess(diff, 1e-3)


    def test_equivariance_multi_stream(self):
        b, s, h, w, d = 3, 5, 20, 20, 20

        rep1 = (2, )
        rep2 = (1, 2, 3)

        cconv1 = HConv3d(rep1, rep2, s).double()
        cconv2 = HConv3d(rep2, rep1, s).double()

        inp = torch.randn(2, b, rep1[0], h, w, d, dtype=torch.float64)
        rot = rot90(inp)

        base_fwd = cconv2(cconv1(inp))
        rot_fwd = cconv2(cconv1(rot))

        diff = (rot90(base_fwd) - rot_fwd).max().item()
        
        self.assertLess(diff, 1e-3)


    def test_equivariance_multi_stream_two_hops(self):
        b, s, h, w, d = 3, 5, 20, 20, 20

        rep1 = (2, )
        rep2 = (1, 2, 3)
        rep3 = (4, 5, 6)
        rep4 = (2, )

        cconv1 = HConv3d(rep1, rep2, s).double()
        cconv2 = HConv3d(rep2, rep3, s).double()
        cconv3 = HConv3d(rep3, rep4, s).double()

        inp = torch.randn(2, b, rep1[0], h, w, d, dtype=torch.float64)
        rot = rot90(inp)

        base_fwd = cconv3(cconv2(cconv1(inp)))
        rot_fwd = cconv3(cconv2(cconv1(rot)))

        diff = (rot90(base_fwd) - rot_fwd).max().item()
        
        self.assertLess(diff, 1e-3)

    def test_equivariance_multi_stream_two_hops_sparse(self):
        b, s, h, w, d = 3, 5, 20, 20, 20

        rep1 = (2, )
        rep2 = (1, 0, 3)
        rep3 = (0, 5, 6)
        rep4 = (2, )

        cconv1 = HConv3d(rep1, rep2, s).double()
        cconv2 = HConv3d(rep2, rep3, s).double()
        cconv3 = HConv3d(rep3, rep4, s).double()

        inp = torch.randn(2, b, rep1[0], h, w, d, dtype=torch.float64)
        rot = rot90(inp)

        base_fwd = cconv3(cconv2(cconv1(inp)))
        rot_fwd = cconv3(cconv2(cconv1(rot)))

        diff = (rot90(base_fwd) - rot_fwd).max().item()
        
        self.assertLess(diff, 1e-3)

unittest.main()
