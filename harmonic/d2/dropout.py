import torch

class Dropout2d(torch.jit.ScriptModule):
    __constants__ = ['p']

    def __init__(self, p=0.1):
        if not 0. < p < 1.:
            raise ValueError(f"`p` must be between 0. and 1., got {p}")

        super(Dropout2d, self).__init__()
        self.p = p

    @torch.jit.script_method
    def forward(self, x):
        if self.training:
            sample = torch.rand(
                (1, 1, x.size(2), 1, 1),
                dtype=torch.float32, device=x.device
            )

            should_drop = (sample > self.p).to(torch.float32)
            result = x * should_drop / (1. - self.p)
        else:
            result = x

        return result

    def __repr__(self):
        return f'Dropout2d(p={self.p})'

if __name__ == '__main__':
    data = torch.randn(2, 1, 30, 2, 2) 
    d = Dropout2d()
    o = d(data)
    print(o)
