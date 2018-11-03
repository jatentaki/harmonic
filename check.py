import torch
import matplotlib.pyplot as plt

from hnet import HNet

layout = [
    (1, ),
    (5, 5, 5),
    (5, 5, 5),
    (5, 5, 5),
    (10, ),
]

net = HNet(3, layout=layout)

load = torch.load('e16.pth.tar', map_location='cpu')

net.load_state_dict(load['model'])

cconv0 = net.seq[2].conv

k = cconv0.convs['1_0'].weights.synthesize().detach().numpy()

def show_kernel(k):
    ins = k.shape[0]
    outs = k.shape[1]

    fig_r, axes_r = plt.subplots(ins, outs)
    fig_i, axes_i = plt.subplots(ins, outs)

    for i in range(ins):
        for o in range(outs):
            axes_i[i, o].imshow(k[i, o, :, :, 1])
            axes_r[i, o].imshow(k[i, o, :, :, 0])

    plt.show()

show_kernel(k)
