from torch_dimcheck import dimchecked

from ..cat import catnd

@dimchecked
def cat2d(t1: [2, 'b', 'f1', 'h', 'w'], repr1,
          t2: [2, 'b', 'f2', 'h', 'w'], repr2) -> [2, 'b', 'fo', 'h', 'w']:
    '''
    concatenate 2d tensor `t1`, containing representations `repr1` with 2d tensor
    `t2` containing representations `repr2`. Returns 2d tensor `t3`
    '''

    return catnd(t1, repr1, t2, repr2)
