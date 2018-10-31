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
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(net.parameters(), lr=0.01)

ixs = torch.arange(10000)

for epoch in range(20):
    random.shuffle(ixs)
    for batch in range(100):
        sample = ixs[100*batch:100*(batch+1)]
        maps = net(train_x[sample, ...])
        predictions = maps.sum(dim=(2, 3))
        optim.zero_grad()
        loss = loss_fn(predictions, train_y[sample, ...])
        print('Loss', loss.item())
        loss.backward()
        optim.step()
