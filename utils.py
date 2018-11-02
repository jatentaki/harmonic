class AvgMeter:
    def __init__(self):
        self.n = 0
        self.avg = 0
        self.last = None

    def update(self, val):
        self.last = val
        self.avg = val / (self.n + 1) + self.avg * self.n / (self.n + 1)
        self.n += 1

    def reset(self):
        self.n = 0
        self.avg = 0
        self.last = None

def rot90(x, k=1, plane=(2, 3)):
    if k == 0:
        return x
    elif k == 1:
        return x.flip(plane[0]).transpose(*plane)
    elif k == 2:
        return x.flip(plane[0]).flip(plane[1])
    elif k == 3:
        return x.flip(plane[1]).transpose(*plane)
    else:
        raise ValueError("k={} is invalid".format(k))
