import torch
import torch.nn.functional as F
from torch_dimcheck import dimchecked

@dimchecked
def magnitude_sqr(t: [2, ...]) -> [...]:
    return t.pow(2).sum(dim=0)


@dimchecked
def magnitude(t: [2, ...], eps=1e-8) -> [...]:
    return torch.sqrt(magnitude_sqr(t) + eps)


@dimchecked
def cmplx(real: [...], imag: [...]) -> [2, ...]:
    return torch.stack([real, imag], dim=0)


@dimchecked
def from_real(real: [...]) -> [2, ...]:
    return cmplx(real, torch.zeros_like(real))


@dimchecked
def conv_nd(x: [2, 'b',     'f_in', 'hx', 'wx', ...,],
            w: [2, 'f_out', 'f_in', 'hk', 'wk', ...,],
            dim=2, pad=False) -> [2, 'b', 'f_out', 'ho', 'wo', ...]:

    if dim not in [2, 3]:
        raise ValueError("Dim can only be 2 or 3, got {}".format(dim))

    if pad:
        padding = w.shape[3] // 2
    else:
        padding = 0

    conv = F.conv3d if dim == 3 else F.conv2d

    real = conv(x[0, ...], w[0, ...], padding=padding) - \
           conv(x[1, ...], w[1, ...], padding=padding)

    imag = conv(x[0, ...], w[1, ...], padding=padding) + \
           conv(x[1, ...], w[0, ...], padding=padding)

    return cmplx(real, imag)
