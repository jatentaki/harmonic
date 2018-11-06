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

@dimchecked
def upsample_3d(input: ['n', 'f', 'hi', 'wi', 'di', 2],
                scale_factor=2, align_corners=False
               ) -> ['n', 'f', 'ho', 'wo', 'do', 2]:
    '''
        Spatial trilinear upsampling without mixing real and imaginary parts
    '''

    real = input[..., 0]
    imag = input[..., 1]

    return cmplx(
        F.interpolate(
            real, scale_factor=scale_factor,
            mode='trilinear', align_corners=align_corners
        ),
        F.interpolate(
            imag, scale_factor=scale_factor,
            mode='trilinear', align_corners=align_corners
        )
    )

