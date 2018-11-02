import torch
import numpy as np
import torchvision as tv
import torchvision.transforms as T

class Rotmnist(torch.utils.data.DataLoader):
    def __init__(self, path, transform=T.Compose([])):
        data = np.load(path)
        self.transform = transform

        x = torch.from_numpy(data['x'])
        y = torch.from_numpy(data['y'])

        self.x = x.reshape(-1, 1, 28, 28)
        self.y = y.long()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]

        return self.transform(x), y
