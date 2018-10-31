import torch
import torch.nn as nn
from torch_dimcheck import ShapeChecker

def magnitude(t: [..., 2]) -> [...]:
    return (t ** 2).sum(dim=-1) ** 0.5

class ScalarGate(nn.Module):
    def __init__(self, repr):
        super(ScalarGate, self).__init__()

        self.repr = repr

        total_fmaps = sum(repr)
        self.conv1 = nn.Conv2d(total_fmaps, total_fmaps, 1)
        self.conv2 = nn.Conv2d(total_fmaps, total_fmaps, 1)

    def forward(self, *streams):
        if len(streams) != len(self.repr):
            fmt = "Based on repr {} expected {} streams, got {}"
            msg = fmt.format(self.repr, len(self.repr), len(streams))
            raise ValueError(msg)

        checker = ShapeChecker()
        for i, stream in enumerate(streams):
            if stream is None:
                continue

            checker.check(stream, ['n', -1, 'h', 'w', 2], name='in_stream {}'.format(i))

        magnitudes = [magnitude(s) for s in streams if s is not None]
        # cat along feature axis to gate across all streams
        magnitudes = torch.cat(magnitudes, dim=1) 
        
        g = self.conv1(magnitudes)
        g = torch.relu(g)
        g = self.conv2(g)
        g = torch.sigmoid(g)

        out_streams = []
        offset = 0
        for mult, stream in zip(self.repr, streams):
            if stream is None:
                out_streams.append(None)
                continue

            out_streams.append(g[:, offset:offset+mult, ...].unsqueeze(-1) * stream)
            offset += mult

        for i, stream in enumerate(out_streams):
            if stream is None:
                continue

            checker.check(stream, ['n', -1, 'h', 'w', 2], name='out_stream {}'.format(i))

        return out_streams
            
