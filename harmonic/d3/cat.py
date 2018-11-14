from torch_dimcheck import dimchecked

from ..cat import catnd

@dimchecked
def cat3d(t1: [2, 'b', 'f1', 'h', 'w', 'd'], repr1,
          t2: [2, 'b', 'f2', 'h', 'w', 'd'], repr2) -> [2, 'b', 'fo', 'h', 'w', 'd']:
    '''
    concatenate 3d tensor `t1`, containing representations `repr1` with 3d tensor
    `t2` containing representations `repr2`. Returns 3d tensor `t3`
    '''

    return catnd(t1, repr1, t2, repr2)
