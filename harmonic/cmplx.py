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
            dim=2, transpose=False, hermitian=False,
            **kwargs) -> [2, 'b', 'f_out', 'ho', 'wo', ...]:

    if dim not in [2, 3]:
        raise ValueError("Dim can only be 2 or 3, got {}".format(dim))

    options = {
        (2, False): F.conv2d,
        (3, False): F.conv3d,
        (2, True): F.conv_transpose2d,
        (3, True): F.conv_transpose3d
    }
    conv = options[(dim, transpose)]

    if transpose or hermitian:
        w = w.transpose(1, 2)
    if hermitian:
        w = torch.flip(w, [0])

    real = conv(x[0, ...], w[0, ...], **kwargs) - \
           conv(x[1, ...], w[1, ...], **kwargs)

    imag = conv(x[0, ...], w[1, ...], **kwargs) + \
           conv(x[1, ...], w[0, ...], **kwargs)

    return cmplx(real, imag)
