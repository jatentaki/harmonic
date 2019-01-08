import torch, itertools, math
import torch.nn as nn
from torch_localize import localized_module
from torch_dimcheck import dimchecked

from .cmplx import cmplx, conv_nd
from .weights import Weights


def ords2s(in_ord, out_ord):
    return '{}_{}'.format(in_ord, out_ord)


@localized_module
class _HConv(nn.Module):
    def __init__(self, in_repr, out_repr, size, radius=None, dim=2,
                 transpose=False, conv_kwargs=dict()):
        super(_HConv, self).__init__()

        if dim not in [2, 3]:
            raise ValueError("Dim can only be 2 or 3, got {}".format(dim))

        self.dim = dim
        self.in_repr = in_repr
        self.out_repr = out_repr
        self.size = size
        self.transpose = transpose
        self.conv_kwargs = conv_kwargs

        self.constrained = True
        self.conv = _HConvConstr(in_repr, out_repr, size, radius=radius, dim=dim)

        self.radius = self.conv.radius


    def relax(self):
        '''
        remove the constraints on convolution kernels, essentially replacing the
        layer with a regular 2/3d convolution computed on complex numbers
        and initialized to the current state of this layer
        '''
        
        if not self.constrained:
            raise ValueError("Attempting to relax a relaxed HConv")

        conv = _RelaxedHConv(self.conv)
        del self.conv
        self.conv = conv
        self.constrained = False

        return self

    def __repr__(self):
        return (f'HConv{self.dim}d(in={self.repr_in}, out={self.repr_out}, '
                f'size={self.size}, radius={self.size}, constr={self.constrained}, '
                f'trans={self.transpose}, {self.conv_kwargs}')


    def forward(self, x: [2, 'b', 'fi', 'hx', 'wx', ...]
               ) -> [2, 'b', 'fo', 'ho', 'wo', ...]:
        if x.shape[2] != sum(self.in_repr):
            fmt = "Based on repr {} expected {} feature maps, found {}"
            msg = fmt.format(self.in_repr, sum(self.in_repr), x.shape[2])
            raise ValueError(msg)

        return conv_nd(
            x, self.conv.synthesize(), dim=self.dim,
            transpose=self.transpose, **self.conv_kwargs
        )
        

@localized_module
class _HConvConstr(nn.Module):
    def __init__(self, in_repr, out_repr, size, radius=None, dim=2):
        super(_HConvConstr, self).__init__()

        self.dim = dim
        self.in_repr = in_repr
        self.out_repr = out_repr
        self.size = size

        self.radius = radius if radius is not None else size / 2 - 0.5
        self.weights = nn.ModuleDict()

        # 2d convolutions take features_in feature maps as input,
        # 3d take features_in * size feature maps (features_in times
        # size of z-dimension)
        mul = 1 if dim == 2 else size

        # create Weights for each (input order, output order) pair
        for (in_ord, in_mult), (out_ord, out_mult) in itertools.product(
                                    enumerate(in_repr),
                                    enumerate(out_repr)):

            if in_mult == 0 or out_mult == 0:
                # either order is not represented in current (in, out) pair
                continue

            name = 'Weights {}x{} -> {}x{}'.format(
                in_mult, in_ord, out_mult, out_ord
            )


            order_diff = in_ord - out_ord
            weight = Weights(
                in_mult * mul, out_mult, size, self.radius, order_diff, name=name
            )
            self.weights[ords2s(in_ord, out_ord)] = weight 

    def __repr__(self):
        fmt = '_HConvConstr{}d(repr_in={}, repr_out={}, size={}, radius={})'
        msg = fmt.format(
            self.dim, self.in_repr, self.out_repr, self.size, self.radius
        )
        return msg


    def synthesize(self) -> [2, 'fo', 'fi', 'h', 'w', ...]:
        spatial_unsqueeze = [self.size] * self.dim

        input_kernels = []
        for (in_ord, in_mult) in enumerate(self.in_repr):
            output_kernels = []
            if in_mult == 0:
                continue

            for (out_ord, out_mult) in enumerate(self.out_repr):
                if out_mult == 0:
                    continue

                ix = ords2s(in_ord, out_ord)
                kernel = self.weights[ix].cartesian_harmonics()
                kernel = kernel.reshape(
                    2, out_mult, in_mult, *spatial_unsqueeze
                )

                output_kernels.append(kernel)

            input_kernels.append(torch.cat(output_kernels, dim=1))

        return torch.cat(input_kernels, dim=2)
        

@localized_module
class _RelaxedHConv(nn.Module):
    def __init__(self, conv_constr):
        super(_RelaxedHConv, self).__init__()

        kernel = conv_constr.synthesize()
        self.kernel = nn.Parameter(conv_constr.synthesize().detach())

        self.dim = conv_constr.dim
        self.in_repr = conv_constr.in_repr
        self.out_repr = conv_constr.out_repr
        self.size = conv_constr.size
        self.radius = conv_constr.radius

    def synthesize(self) -> [2, 'fo', 'fi', 'h', 'w', ...]:
        return self.kernel

    def __repr__(self):
        fmt = '_RelaxedHConv{}d(repr_in={}, repr_out={}, size={}, radius={})'
        msg = fmt.format(
            self.dim, self.in_repr, self.out_repr, self.size, self.radius
        )
        return msg
