import torch, unittest
import conv

class WeightsTests(unittest.TestCase):
    def test_radial(self):
        f_in, f_out = 2, 3
        rad = 5
        diam = 2 * rad + 1

        weights = conv.Weights(f_in, f_out, rad, 0)

        xs = torch.linspace(-rad, rad, diam).reshape(-1, 1)
        ys = xs.reshape(1, -1)
        rs = torch.sqrt(xs ** 2 + ys ** 2).reshape(1, diam, diam)

        expected = torch.ones_like(rs)
        expected[rs > rad] = 0

        result = weights.radial()
        self.assertTrue(torch.allclose(result, expected))


class HConvTests(unittest.TestCase):
    def test_no_crash(self):
        hconv = conv.HConv(5, 10, 5, 2, pad=False)
        
        input = torch.randn(2, 5, 20, 20, 2, requires_grad=True)
        output = hconv(input)


class CrossConvTests(unittest.TestCase):
    def test_streams(self):
        cconv = conv.CrossConv((1, 2), (3, 1), 4, pad=True)
        n, h, w = 3, 40, 40
        input = [
            torch.randn(n, 1, h, w, 2),
            torch.randn(n, 2, h, w, 2)
        ]

        out1, out2 = cconv(*input)

        self.assertEqual(out1.shape, (n, 3, h, w, 2))
        self.assertEqual(out2.shape, (n, 1, h, w, 2))

unittest.main()
