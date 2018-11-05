import torch
import torch.nn.functional as F

from torch_dimcheck import dimchecked

from ..cmplx import cmplx 

@dimchecked
def avg_pool3d(input: ['n', 'f', 'hi', 'wi', 'di', 2],
               *args, **kwargs) -> ['n', 'f', 'ho', 'wo', 'do', 2]:
    '''
        Do spatial average pooling without mixing real and imaginary parts
    '''

    real = input[..., 0]
    imag = input[..., 1]

    return cmplx(
        F.avg_pool3d(real, *args, **kwargs),
        F.avg_pool3d(imag, *args, **kwargs)
    )
