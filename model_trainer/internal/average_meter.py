class AverageMeter:
    """
    Compute average and current value.
    """
    def __init__(self, name="Meter", fmt="3f", avg_mom=0.9):
        """

        :param name:
        :param avg_mom:
        """
        self.avg_mom = avg_mom
        self.name = name

        self.count = None
        self.avg_smooth = None
        self.last_value = None
        self.avg = None

        self.reset()

    def reset(self):
        """
        :return:
        """
        self.last_value = 0
        self.avg_smooth = 0
        self.count = 0
        self.avg = 0

    def update(self, val):
        """
        Each update, update last value and recompute average.
        :param val:
        :return:
        """
        self.last_value = val
        if self.count == 0:
            self.avg_smooth = val
        else:
            self.avg_smooth *= self.avg_mom
            self.avg_smooth += val * (1 - self.avg_mom)
        self.count += 1
        self.avg *= (self.count - 1) / self.count
        self.avg += val / self.count

    def __call__(self, val):
        return self.update(val)

    def __repr__(self):
        return f"Average Meter(name={self.name}, avg={self.avg:.3f}, count={self.count})"
