import torch, random, imageio, os
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from hnet import HNet
from loader import Rotmnist

mean = 0.13
std = 0.3

train_loader = torch.utils.data.DataLoader(
    Rotmnist('rotmnist/rotated_train.npz', transform=T.Normalize((mean, ), (std, ))),
    batch_size=250, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    Rotmnist('rotmnist/rotated_test.npz', transform=T.Normalize((mean, ), (std, ))),
    batch_size=1000, shuffle=False
)

layout = [
    (1, ),
    (3, 5, 3),
    (3, 5, 3),
    (10, ),
]

net = HNet(3, layout=layout)
loss_fn = nn.CrossEntropyLoss()

cuda = torch.cuda.is_available()
if cuda:
    net = net.cuda()
    loss_fn = loss_fn.cuda()

n_params = 0
for param in net.parameters():
    n_params += param.numel()
print('n params:', n_params)

optim = torch.optim.Adam(net.parameters())

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

def unnormalize(x):
    return x * std + mean

train_ixs = torch.arange(10000)
test_ixs = torch.arange(2000)

n_epochs = 20

savedir = 'saves/'
save_nr = 0

def save_one(x, predictions, epoch, batch, prefix=''):
    global save_nr
    img = x[0, 0, ...].detach().cpu().numpy()
    _, pred = predictions[0].topk(1)

    path = savedir + '/{}/e{}/'.format(prefix, epoch)
    if not os.path.isdir(path):
        os.makedirs(path)

    fname = path + 'p{}_{}.png'.format(pred.item(), save_nr)
    imageio.imsave(
        fname,
        (unnormalize(img) * 255).astype(np.uint8)
    )

    save_nr += 1
    
for epoch in range(n_epochs):
    for i, (x, y) in enumerate(train_loader):
        if cuda:
            x, y = x.cuda(), y.cuda()

        maps = net(x)
        predictions = maps.sum(dim=(2, 3))
        optim.zero_grad()
        loss = loss_fn(predictions, y)
        acc = accuracy(predictions, y)
        fmt = 'Train\tLoss: {:.2f}\tAccuracy {:.2f} out of {}'
        print(fmt.format(loss.item(), acc[0].item(), y.shape[0]))
        loss.backward()
        optim.step()
        save_one(x, predictions, epoch, i, prefix='train')

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if cuda:
                x, y = x.cuda(), y.cuda()

            maps = net(x)
            predictions = maps.sum(dim=(2, 3))
            acc = accuracy(predictions, y)
            fmt = 'Test\tAccuracy {:.2f} out of {}'
            print(fmt.format(acc[0].item(), y.shape[0]))
            save_one(x, predictions, epoch, i, prefix='test')
