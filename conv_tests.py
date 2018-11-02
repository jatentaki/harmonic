import torch, unittest
from conv import HConv, CrossConv
from utils import rot90
from cmplx import magnitude

class HConvTests(unittest.TestCase):
    def test_equivariance(self):
        b, r, c1, c2, h, w = 5, 7, 5, 10, 30, 30
        conv1 = HConv(c1, c2, r, 2).double()
        conv2 = HConv(c2, c1, r, -2).double()

        inp = torch.randn(b, c1, h, w, 2, dtype=torch.float64)
        rot = rot90(inp)

        base_fwd = conv2(conv1(inp))
        rot_fwd = conv2(conv1(rot))

        diff = (magnitude(rot90(base_fwd)) - magnitude(rot_fwd)).max().item()
        
        self.assertLess(diff, 1e-3)

class CrossConvTests(unittest.TestCase):
    def test_streams(self):
        r = 7
        cconv = CrossConv((1, 2), (3, 1), r, pad=True)
        n, h, w = 3, 40, 40
        input = [
            torch.randn(n, 1, h, w, 2),
            torch.randn(n, 2, h, w, 2)
        ]

        out1, out2 = cconv(*input)

        self.assertEqual(out1.shape, (n, 3, h, w, 2))
        self.assertEqual(out2.shape, (n, 1, h, w, 2))
    
    def test_equivariance(self):
        b, r, h, w = 5, 7, 50, 50

        rep1 = (2, )
        rep2 = (3, 3, 3)

        cconv1 = CrossConv(rep1, rep2, r)
        cconv2 = CrossConv(rep2, rep1, r)

        inp = torch.randn(b, rep1[0], h, w, 2)
        rot = rot90(inp)

        base_fwd = cconv2(*cconv1(inp))
        rot_fwd = cconv2(*cconv1(rot))

        # single stream
        self.assertEqual(len(base_fwd), 1)
        self.assertEqual(len(rot_fwd), 1)

        diff = (magnitude(base_fwd[0]) - magnitude(rot90(rot_fwd[0], k=3))).max().item()
        
        self.assertLess(diff, 1e-3)

unittest.main()
