import torch, random, imageio, os, argparse
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm

import loader
from hnet import HNet
from baseline import BNet
from utils import AvgMeter


parser = argparse.ArgumentParser()
parser.add_argument('model', choices=['hnet', 'baseline'])
parser.add_argument('train', choices=['mnist', 'rotmnist'])
parser.add_argument('-b', '--batch', type=int, default=250)
parser.add_argument('-j', '--workers', type=int, default=1)

args = parser.parse_args()

normalize = T.Normalize((0.1307,), (0.3081,))
def unnormalize(x):
    return x * .3081 + .1307

if args.train == 'mnist':
    train_set = tv.datasets.MNIST(
        '../data', train=True, download=True,
        transform=T.Compose([
            T.ToTensor(),
            normalize
        ])
    )
elif args.train == 'rotmnist':
    train_set = loader.Rotmnist('rotmnist/rotated_train.npz', transform=normalize)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers
)

test_loader = torch.utils.data.DataLoader(
    loader.Rotmnist('rotmnist/rotated_test.npz', transform=normalize),
    batch_size=args.batch, shuffle=False
)

layout = [
    (1, ),
    (2, 5, 3, 3, 2),
#    (5, 5, 5),
    (5, 5, 5),
    (5, 5, 5),
    (10, ),
]

if args.model == 'hnet':
    net = HNet(5, layout=layout)
elif args.model == 'baseline':
    net = BNet(5, layout=layout)

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
    
print('tracing...')
example = next(iter(train_loader))[0]
if cuda:
    example = example.cuda()
net = torch.jit.trace(net, example)
print('done')

for epoch in range(n_epochs):
    with tqdm(total=len(train_loader), dynamic_ncols=True) as progress:
        mean_loss = AvgMeter()
        mean_acc = AvgMeter()
        for i, (x, y) in enumerate(train_loader):
#            x, y = x.clone(), y.clone()
            if cuda:
                x, y = x.cuda(), y.cuda()

            optim.zero_grad()
            maps = net(x)
            predictions = maps.sum(dim=(2, 3))
            loss = loss_fn(predictions, y)
            loss.backward()

            acc = accuracy(predictions, y)
            optim.step()
            progress.update(1)
            mean_loss.update(loss.item())
            mean_acc.update(acc[0].item())
            progress.set_postfix(loss=mean_loss.avg, accuracy=mean_acc.avg)
            save_one(x, predictions, epoch, i, prefix='train')

    with torch.no_grad(), tqdm(total=len(test_loader), dynamic_ncols=True) as progress:
        mean_acc = AvgMeter()
        for i, (x, y) in enumerate(test_loader):
#            x, y = x.clone(), y.clone()
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
