import torch
from torch_dimcheck import dimchecked

@dimchecked
def magnitude(t: [..., 2], eps=1e-8) -> [...]:
    return torch.sqrt(t.pow(2).sum(dim=-1) + eps)

@dimchecked
def cmplx(real: [...], imag: [...]) -> [..., 2]:
    return torch.stack([real, imag], dim=-1)

@dimchecked
def from_real(real: [...]) -> [..., 2]:
    return cmplx(real, torch.zeros_like(real))
