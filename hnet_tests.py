import torch, unittest
from hnet import CrossConv, HNet

class HNetTests(unittest.TestCase):
    def test_instantiation(self):
        hnet = HNet(4)

    def test_forward(self):
        hnet = HNet(4)
        input = torch.randn(7, 1, 60, 60)

        output = hnet(input)


unittest.main()
