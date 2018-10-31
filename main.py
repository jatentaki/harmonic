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
    (5, 5, 5),
    (3, 5, 5),
    (10, ),
]

net = HNet(4, layout=layout)
n_params = 0
for param in net.parameters():
    n_params += param.numel()
print('n params:', n_params)

loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(net.parameters(), lr=0.01)

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

train_ixs = torch.arange(10000)
test_ixs = torch.arange(2000)

n_epochs = 20
batch_size = 100
n_train = 250
n_test = 100

for epoch in range(n_epochs):
    random.shuffle(train_ixs)
    for batch in range(n_train // batch_size):
        sample = train_ixs[batch_size*batch:batch_size*(batch+1)]
        maps = net(train_x[sample, ...])
        predictions = maps.sum(dim=(2, 3))
        optim.zero_grad()
        loss = loss_fn(predictions, train_y[sample, ...])
        print('Loss', loss.item())
        loss.backward()
        print(predictions.grad)
        optim.step()

    random.shuffle(test_ixs)
    for batch in range(n_test // batch_size):
        sample = test_ixs[batch_size*batch:batch_size*(batch+1)]
        maps = net(test_x[sample, ...])
        predictions = maps.sum(dim=(2, 3))
        acc = accuracy(predictions, test_y[sample, ...])
        print('accuracy', acc[0].item(), 'out of', sample.shape[0])
