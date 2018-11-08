import torch, unittest
from hnet import HNet
from utils import rot90

class HNetTests(unittest.TestCase):
    def test_equivariance(self):
        hnet = HNet(5)
        inp = torch.randn(7, 1, 60, 60)
        inp_rot = rot90(inp)

        out = hnet(inp)
        out_rot = hnet(inp_rot)

        diff = (rot90(out) - out_rot).max().item()

        self.assertLess(diff, 1e-3)

    def test_jitable(self):
        hnet = HNet(5)
        inp = torch.randn(2, 1, 60, 60)

        jitted = torch.jit.trace(hnet, inp)
        
#    def test_grad(self):
#        hnet = HNet(5)
#        inp = torch.randn(2, 1, 60, 60).requires_grad_(True)
#
#        f = lambda i: hnet(i).sum(dim=(2, 3))
#        torch.autograd.gradcheck(f, (inp, ))

unittest.main()
