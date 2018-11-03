import torch
from torch_dimcheck import dimchecked

@dimchecked
def magnitude(t: [..., 2]) -> [...]:
    return (t ** 2).sum(dim=-1) ** 0.5

@dimchecked
def cmplx(real: [...], imag: [...]) -> [..., 2]:
    return torch.stack([real, imag], dim=-1)
