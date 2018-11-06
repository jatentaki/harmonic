import torch
import torch.nn.functional as F

from torch_dimcheck import dimchecked

from ..cmplx import cmplx 

@dimchecked
def avg_pool2d(input: ['n', 'f', 'hi', 'wi', 2],
               *args, **kwargs) -> ['n', 'f', 'ho', 'wo', 2]:
    '''
        Spatial average pooling without mixing real and imaginary parts
    '''

    real = input[..., 0]
    imag = input[..., 1]

    return cmplx(
        F.avg_pool2d(real, *args, **kwargs),
        F.avg_pool2d(imag, *args, **kwargs)
    )

@dimchecked
def upsample_2d(input: ['n', 'f', 'hi', 'wi', 2],
                scale_factor=2, align_corners=False
               ) -> ['n', 'f', 'ho', 'wo', 2]:
    '''
        Spatial bilinear upsampling without mixing real and imaginary parts
    '''

    real = input[..., 0]
    imag = input[..., 1]

    return cmplx(
        F.interpolate(
            real, scale_factor=scale_factor,
            mode='bilinear', align_corners=align_corners
        ),
        F.interpolate(
            imag, scale_factor=scale_factor,
            mode='bilinear', align_corners=align_corners
        )
    )

