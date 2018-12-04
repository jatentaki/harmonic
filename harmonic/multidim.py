import torch
from functools import partial
from collections.abc import Sequence

def _reduce_many(tensor, dim, keepdim=False, operation='std', **kwargs):
    '''
        executed reduction method `operation` on tensor `tensor` allowing
        multi-dimensional reductions even if the base implementation of
        `operation` allows only a single reduction dimension
    '''
    if not (isinstance(dim, Sequence) and all(isinstance(d, int) for d in dim)):
        return getattr(tensor, operation)(dim, keepdim=keepdim, **kwargs)

    def to_positive(ix):
        return ix if ix >= 0 else tensor.dim() + ix

    to_reduce = [to_positive(d) for d in dim]
    to_keep = [i for i in range(tensor.dim()) if i not in to_reduce]

    permuted = tensor.permute(*to_keep, *to_reduce)
    reshaped = permuted.reshape(*[tensor.shape[i] for i in to_keep], -1)
    reduced = getattr(reshaped, operation)(dim=-1, **kwargs)

    if not keepdim:
        return reduced

    shape = [(tensor.shape[i] if i in to_keep else 1) for i in range(tensor.dim())]
    return reduced.reshape(*shape)

std = partial(_reduce_many, operation='std')
mean = partial(_reduce_many, operation='mean')
max = partial(_reduce_many, operation='max')
min = partial(_reduce_many, operation='min')
