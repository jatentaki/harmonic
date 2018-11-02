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
