import torch, unittest
from hnet import CrossConv, HNet

class HNetTests(unittest.TestCase):
    def test_instantiation(self):
        hnet = HNet(4)

    def test_forward(self):
        hnet = HNet(4)
        print(repr(hnet))
        input = torch.randn(3, 1, 60, 60)

        output = hnet(input)

class CrossConvTests(unittest.TestCase):
    def test_streams(self):
        cconv = CrossConv((1, 2), (3, 1), 4, pad=True)
        n, h, w = 3, 40, 40
        input = [
            torch.randn(n, 1, h, w, 2),
            torch.randn(n, 2, h, w, 2)
        ]

        out1, out2 = cconv(*input)

        self.assertEqual(out1.shape, (n, 3, h, w, 2))
        self.assertEqual(out2.shape, (n, 1, h, w, 2))


unittest.main()
