import time
import math


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


class Timer(object):
    def __init__(self):
        super().__init__()
        self.last_time = time.time()

    def time_difference(self):
        current_time = time.time()
        difference = current_time - self.last_time
        self.last_time = current_time
        return difference
