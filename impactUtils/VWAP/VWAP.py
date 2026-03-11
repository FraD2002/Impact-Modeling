# Class to calculate volume weighted average
# price for a given day of data between
# start timestamp and end timesamp,
# exclusive
class VWAP(object):
    def __init__(self, data, startTS, endTS):
        if endTS <= startTS:
            raise ValueError("endTS must be greater than startTS.")

        v = 0.0
        s = 0.0
        counter = 0
        for i in range(0, data.getN()):
            if data.getTimestamp(i) < startTS:
                continue
            if data.getTimestamp(i) >= endTS:
                break
            size = float(data.getSize(i))
            price = float(data.getPrice(i))
            if size <= 0:
                continue
            counter += 1
            v += size * price
            s += size
        if counter == 0 or s <= 0:
            self._counter = 0
            self._vwap = 0.0
        else:
            self._counter = counter
            self._vwap = v / s

    def getVWAP(self):
        return self._vwap

    def getN(self):
        return self._counter
