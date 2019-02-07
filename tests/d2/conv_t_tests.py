import torch, unittest
from harmonic.d2.conv import HConv2d, HConv2dTranspose
from harmonic.cmplx import magnitude
from utils import rot90

def expand_twice(t):
    t2 = torch.zeros(
        *t.shape[:-2], 2*t.shape[-2], 2*t.shape[-1], dtype=t.dtype
    )

    t2[...,  ::2,  ::2] = t
    t2[..., 1::2,  ::2] = t
    t2[...,  ::2, 1::2] = t
    t2[..., 1::2, 1::2] = t
    
    return t2

class HConvTransposeTests(unittest.TestCase):
    def test_equivariance_0(self):
        self._test_equivariance_transition(0)

    def test_equivariance_1(self):
        self._test_equivariance_transition(1)

    def test_equivariance_2(self):
        self._test_equivariance_transition(2)

    def test_equivariance_3(self):
        self._test_equivariance_transition(3)

    def test_equivariance_4(self):
        self._test_equivariance_transition(4)

    def test_equivariance_m1(self):
        self._test_equivariance_transition(-1)

    def test_equivariance_m2(self):
        self._test_equivariance_transition(-2)

    def test_equivariance_m3(self):
        self._test_equivariance_transition(-3)

    def test_equivariance_m4(self):
        self._test_equivariance_transition(-4)

    def _test_equivariance_transition(self, order):
        b, s, c1, c2, h, w = 5, 7, 5, 10, 32, 32
        repr1 = [c1]
        repr2 = [0] * (order - 1) + [c2]

        conv1 = HConv2d(repr1, repr2, s, conv_kwargs={
            'stride': 2,
        }).double()
#        conv2 = HConv2d(repr2, repr1, s, conv_kwargs={
#            'stride': 2,
#        }).double()
        conv2 = HConv2dTranspose(repr2, repr1, s, conv_kwargs={
            'stride': 2,
            #'output_padding': 1,
        }).double()

        inp = torch.randn(2, b, c1, h, w, dtype=torch.float64)
        inp = expand_twice(inp)
        rot = rot90(inp)

        base_fwd = rot90(conv2(conv1(inp)))
        rot_fwd = conv2(conv1(rot))
        
#        self.assertEqual(inp.shape, base_fwd.shape)

        diff = (base_fwd - rot_fwd)
        
        ax.set_aspect('equal')

        rot_fwd_np = rot_fwd.detach().numpy()
        base_fwd_np = base_fwd.detach().numpy()
        diff_np = rot_fwd_np - base_fwd_np
        ax.quiver(rot_fwd_np[0, 0, 0], base_fwd_np[0, 0, 0])

        
        self.assertLess(diff.max().item(), 1e-5)

unittest.main()
