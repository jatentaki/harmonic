import torch
from torch_dimcheck import dimchecked 

@dimchecked
def catnd(t1: [2, 'b', 'f1', 'h', 'w', ...], repr1,
          t2: [2, 'b', 'f2', 'h', 'w', ...], repr2) -> [2, 'b', 'fo', 'h', 'w', ...]:
    '''
    concatenate tensor `t1`, containing representations `repr1` with tensor
    `t2` containing representations `repr2`. Returns tensor `t3` of representation
    `concatenated_representation(repr1, repr2)`
    '''

    fmt = "size of `t{}` at axis 1 ({}) does not match its representation ({}, total {})"
    if t1.size(2) != sum(repr1):
        msg = fmt.format(1, t1.size(2), repr1, sum(repr1))
        raise ValueError(msg)

    if t2.size(2) != sum(repr2):
        msg = fmt.format(2, t2.size(2), repr2, sum(repr2))
        raise ValueError(msg)

    blocks = []
    prev1, prev2 = 0, 0
    for i, (n1, n2) in enumerate(zip(repr1, repr2)):
        block1 = t1[:, :, prev1:prev1+n1, ...]
        block2 = t2[:, :, prev2:prev2+n2, ...]
        prev1 += n1
        prev2 += n2
        blocks.extend([block1, block2])

    # one of representations must have been exhausted
    if prev1 < t1.size(2) and prev2 < t2.size(2):
        msg = "logical error: neither iterator exhausted"
        raise AssertionError(msg)

    if prev1 < t1.size(2):
        blocks.append(t1[:, :, prev1:, ...])
    if prev2 < t2.size(2):
        blocks.append(t2[:, :, prev2:, ...])

    return torch.cat(blocks, dim=2)
