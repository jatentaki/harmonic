import torch, itertools, math
import torch.nn as nn
from torch_localize import localized_module
from torch_dimcheck import dimchecked

from .cmplx import cmplx, conv_nd
from .weights import Weights


def ords2s(in_ord, out_ord):
    return '{}_{}'.format(in_ord, out_ord)


@localized_module
class _HConv_1x1(nn.Module):
    def __init__(self, in_repr, out_repr, dim=2):
        super(_HConv_1x1, self).__init__()

        if dim not in [2, 3]:
            raise ValueError("Dim can only be 2 or 3, got {}".format(dim))

        self.dim = dim
        self.in_repr = in_repr
        self.out_repr = out_repr

        self.constrained = True
        self.conv = _HConvConstr_1x1(in_repr, out_repr, dim=dim)


    def relax(self):
        '''
        remove the constraints on convolution kernels, essentially replacing the
        layer with a regular 2/3d convolution computed on complex numbers
        and initialized to the current state of this layer
        '''
        
        if not self.constrained:
            raise ValueError("Attempting to relax a relaxed HConv")

        conv = _RelaxedHConv_1x1(self.conv)
        del self.conv
        self.conv = conv
        self.constrained = False

        return self

    def __repr__(self):
        fmt = '{}HConv{}d_1x1(in_repr={}, out_repr={}, size=1)'
        msg = fmt.format(
            'Constr' if self.constrained else 'Relaxed', self.dim,
            self.in_repr, self.out_repr,
        )
        return msg


    def forward(self, x: [2, 'b', 'fi', 'hx', 'wx', ...]
               ) -> [2, 'b', 'fo', 'ho', 'wo', ...]:
        if x.shape[2] != sum(self.in_repr):
            fmt = "Based on repr {} expected {} feature maps, found {}"
            msg = fmt.format(self.in_repr, sum(self.in_repr), x.shape[2])
            raise ValueError(msg)

        return self.conv.forward(x)


@localized_module
class Block(nn.Module):
    def __init__(self, out_mul, in_mul, dim=2):
        super(Block, self).__init__()

        self.dim = dim

        trailing = dim * [1]
        conv_params = torch.randn(2, out_mul, in_mul, *trailing)
        self.weights = nn.Parameter(conv_params, requires_grad=True)

    def forward(self, x: [2, 'fi', 'h', 'w', ...]) -> [2, 'fo', 'h', 'w', ...]:
        return conv_nd(x, self.weights, dim=self.dim)

        
@localized_module
class _HConvConstr_1x1(nn.Module):
    def __init__(self, in_repr, out_repr, dim=2):
        super(_HConvConstr_1x1, self).__init__()

        if len(in_repr) != len(out_repr):
            msg = f"Length of in_repr={in_repr} does not match out_repr={out_repr}"
            raise ValueError(msg)

        self.dim = dim
        self.in_repr = in_repr
        self.out_repr = out_repr

        self.blocks = nn.ModuleDict()

        for i, (in_mul, out_mul) in enumerate(zip(in_repr, out_repr)):
            if in_mul != 0 and out_mul == 0:
                msg = (f'Conv1x1 {in_repr} -> {out_repr} '
                       f'maps {in_mul} fmaps of order {i} to 0')
                raise ValueError(msg)

            if in_mul == 0 and out_mul != 0:
                msg = (f'Conv1x1 {in_repr} -> {out_repr} attemps to create '
                       f'{out_mul} fmaps of order {i} out of thin air')
                raise ValueError(msg)

            if in_mul == 0 and out_mul == 0:
                continue

            self.blocks[f'{i}'] = Block(out_mul, in_mul, dim=dim, name=f'block{i}')

    def __repr__(self):
        fmt = '_HConvConstr{}d_1x1(in_repr={}, out_repr={})'
        msg = fmt.format(self.dim, self.in_repr, self.out_repr)
        return msg

    def forward(self, x: [2, 'b', 'f', 'h', 'w', ...]) -> [2, 'b', 'f', 'h', 'w', ...]:
        fmaps = []
        
        s_ix = 0
        for i, mult in enumerate(self.in_repr):
            if mult == 0:
                continue

            maps = x[:, :, s_ix:s_ix+mult, ...]
            block = self.blocks[f'{i}']
            fmaps.append(block(maps))

            s_ix += mult

        return torch.cat(fmaps, dim=2)


@localized_module
class _RelaxedHConv_1x1(nn.Module):
    def __init__(self, conv_constr):
        super(_RelaxedHConv_1x1, self).__init__()

        self.dim = conv_constr.dim
        self.in_repr = conv_constr.in_repr
        self.out_repr = conv_constr.out_repr

        trailing = self.dim * [1]
        example_block = next(iter(conv_constr.blocks.values())).weights
        kernel = torch.zeros(
            2, sum(self.out_repr), sum(self.in_repr), *trailing,
            dtype=example_block.dtype, device=example_block.device
        )

        o_s = 0
        i_s = 0

        for i, (out_mult, in_mult) in enumerate(zip(self.out_repr, self.in_repr)):
            weights = conv_constr.blocks[f'{i}'].weights
            kernel[:, o_s:o_s+out_mult, i_s:i_s+in_mult, ...] = weights

            o_s += out_mult
            i_s += in_mult

        self.kernel = nn.Parameter(kernel.detach(), requires_grad=True)

    def __repr__(self):
        fmt = '_RelaxedHConv{}d_1x1(in_repr={}, out_repr={})'
        msg = fmt.format(
            self.dim, self.in_repr, self.out_repr
        )
        return msg

    def forward(self, x: [2, 'b', 'f', 'h', 'w', ...]) -> [2, 'b', 'f', 'h', 'w', ...]:
        return conv_nd(x, self.kernel, dim=self.dim)
