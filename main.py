import torch, random
import torch.nn as nn
import numpy as np
from hnet import HNet

train = np.load('rotmnist/rotated_train.npz')
test = np.load('rotmnist/rotated_test.npz')

train_x, train_y = torch.from_numpy(train['x']), torch.from_numpy(train['y'])
test_x, test_y = torch.from_numpy(test['x']), torch.from_numpy(test['y'])

train_x = train_x.reshape(-1, 1, 28, 28)
test_x = test_x.reshape(-1, 1, 28, 28)
train_y = train_y.long()
test_y = test_y.long()

layout = [
    (1, ),
    (3, 5, 3),
    (3, 5, 3),
    (10, ),
]

net = HNet(4, layout=layout)
loss_fn = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    net = net.cuda()
    loss_fn = loss_fn.cuda()
    train_x = train_x.cuda()
    test_x = test_x.cuda()
    train_y = train_y.cuda()
    test_y = test_y.cuda()

n_params = 0
for param in net.parameters():
    n_params += param.numel()
print('n params:', n_params)

optim = torch.optim.SGD(net.parameters(), lr=1e-4)

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

# normalize
mean = 0.13
std = 0.3
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

train_ixs = torch.arange(10000)
test_ixs = torch.arange(2000)

n_epochs = 20
batch_size = 250
n_train = 10000
n_test = 2000

for epoch in range(n_epochs):
    random.shuffle(train_ixs)
    for batch in range(n_train // batch_size):
        sample = train_ixs[batch_size*batch:batch_size*(batch+1)]
        maps = net(train_x[sample, ...])
        predictions = maps.sum(dim=(2, 3))
        optim.zero_grad()
        loss = loss_fn(predictions, train_y[sample, ...])
        acc = accuracy(predictions, train_y[sample, ...])
        print('Loss', loss.item(), 'accuracy', acc[0].item(), 'out of', sample.shape[0])
        loss.backward()
        optim.step()

    random.shuffle(test_ixs)
    for batch in range(n_test // batch_size):
        sample = test_ixs[batch_size*batch:batch_size*(batch+1)]
        maps = net(test_x[sample, ...])
        predictions = maps.sum(dim=(2, 3))
        acc = accuracy(predictions, test_y[sample, ...])
        print('accuracy', acc[0].item(), 'out of', sample.shape[0])
