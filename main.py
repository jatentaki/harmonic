import torch, random, imageio, os
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm

from hnet import HNet
from loader import Rotmnist
from utils import AvgMeter

mean = 0.13
std = 0.3

normalize = T.Normalize((0.1307,), (0.3081,))

train_loader = torch.utils.data.DataLoader(
    tv.datasets.MNIST(
        '../data', train=True, download=True,
        transform=T.Compose([
            T.ToTensor(),
            normalize
        ])),
    batch_size=1000, shuffle=True, num_workers=1
)

#train_loader = torch.utils.data.DataLoader(
#    Rotmnist('rotmnist/mnist.npz', transform=T.Normalize((mean, ), (std, ))),
#    batch_size=250, shuffle=True
#)

test_loader = torch.utils.data.DataLoader(
    Rotmnist('rotmnist/rotated_test.npz', transform=normalize),
    batch_size=1000, shuffle=False
)

layout = [
    (1, ),
    (5, 5, 5),
    (5, 5, 5),
    (5, 5, 5),
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
    with tqdm(total=len(train_loader), dynamic_ncols=True) as progress:
        mean_loss = AvgMeter()
        mean_acc = AvgMeter()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.clone(), y.clone()
            if cuda:
                x, y = x.cuda(), y.cuda()

            maps = net(x)
            predictions = maps.sum(dim=(2, 3))
            optim.zero_grad()
            loss = loss_fn(predictions, y)
            acc = accuracy(predictions, y)
            loss.backward()
            optim.step()
            progress.update(1)
            mean_loss.update(loss.item())
            mean_acc.update(acc[0].item())
            progress.set_postfix(loss=mean_loss.avg, accuracy=mean_acc.avg)
            save_one(x, predictions, epoch, i, prefix='train')

    with torch.no_grad(), tqdm(total=len(test_loader), dynamic_ncols=True) as progress:
        mean_acc = AvgMeter()
        for i, (x, y) in enumerate(test_loader):
            x, y = x.clone(), y.clone()
            if cuda:
                x, y = x.cuda(), y.cuda()

            maps = net(x)
            predictions = maps.sum(dim=(2, 3))
            acc = accuracy(predictions, y)
            progress.update(1)
            mean_acc.update(acc[0].item())
            progress.set_postfix(accuracy=mean_acc.avg)
            save_one(x, predictions, epoch, i, prefix='test')

    torch.save({'model': net.state_dict()}, 'e{}.pth.tar'.format(epoch))
