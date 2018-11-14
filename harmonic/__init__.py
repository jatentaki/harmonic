import itertools

import harmonic.d2
import harmonic.d3

def cat_repr(r1, r2):
    '''
    calculate the representation signature of concatenation of signatures
    `r1` and `r2`. For example,
    `cat_repr((3, 5, 1,), (2, 2,)) == (5, 7, 1,)`
    '''

    return tuple(x + y for x, y in itertools.zip_longest(r1, r2, fillvalue=0))
