import torch, unittest
from hnet import CrossConv, HNet

class HNetTests(unittest.TestCase):
    def test_instantiation(self):
        hnet = HNet(4)

unittest.main(failfast=True)
