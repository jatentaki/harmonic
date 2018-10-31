import torch
import conv
import unittest
import matplotlib.pyplot as plt

class WeightsTests(unittest.TestCase):
    def test_radial(self):
        nfmaps = 2
        rad = 5
        diam = 2 * rad + 1

        r = torch.ones(nfmaps, rad + 1, dtype=torch.float32)
        phi = torch.zeros(nfmaps)

        weights = conv.Weights(r, phi, 0)

        xs = torch.linspace(-rad, rad, diam).reshape(-1, 1)
        ys = xs.reshape(1, -1)
        rs = torch.sqrt(xs ** 2 + ys ** 2).reshape(1, diam, diam)

        expected = torch.ones_like(rs)
        expected[rs > rad] = 0

        result = weights.radial()
        self.assertTrue(torch.allclose(result, expected))

    def test_harmonics(self):
        nfmaps = 2
        rad = 5
        diam = 2 * rad + 1

        r = torch.ones(nfmaps, rad + 1, dtype=torch.float32)
        phi = torch.zeros(2)

        weights = conv.Weights(r, phi, 2)

#        plt.imshow(weights.harmonics(phi).real.numpy()[0])
#        plt.figure()
#        plt.imshow(weights.harmonics(phi).imag.numpy()[0])
#        plt.show()

    def test_synthesize(self):
        nfmaps = 2
        rad = 5
        diam = 2 * rad + 1

        r = torch.ones(nfmaps, rad + 1, dtype=torch.float32)
        phi = torch.zeros(2)

        weights = conv.Weights(r, phi, 2)

#        plt.imshow(weights.synthesize().real.numpy()[0])
#        plt.figure()
#        plt.imshow(weights.synthesize().imag.numpy()[0])
#        plt.show()

class HConvTests(unittest.TestCase):
    def test_no_crash(self):
        hconv = conv.HConv(5, 10, 5, 2, pad=False)
        
        input = torch.randn(2, 5, 20, 20, 2, requires_grad=True)
        output = hconv(input)

        plt.imshow(output[0, 0, ..., 0].detach().numpy())
        plt.show()

#    def test_backprop(self):
#        hconv = conv.HConv(5, 10, 5, 2, pad=False)
#        
#        input = torch.randn(2, 5, 20, 20, 2, requires_grad=True)
#        torch.autograd.gradcheck(lambda t: hconv(conv.CTen(t)).t.sum(), (input,), atol=1e-2, rtol=0.01)

unittest.main()
