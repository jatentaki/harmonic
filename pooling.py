import torch
import torch.nn.functional as F

from torch_dimcheck import dimchecked

from cmplx import cmplx 

@dimchecked
def avg_pool2d(input: ['n', 'f', 'hi', 'wi', 2],
               *args, **kwargs) -> ['n', 'f', 'ho', 'wo', 2]:
    '''
        Do spatial average pooling without mixing real and imaginary parts
    '''

    real = input[..., 0]
    imag = input[..., 1]

    return cmplx(
        F.avg_pool2d(real, *args, **kwargs),
        F.avg_pool2d(imag, *args, **kwargs)
    )
