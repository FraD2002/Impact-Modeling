import math

from impactUtils.VWAP import VWAP
from impactUtils.TickTest import TickTest


def getImbalance(trades, startTS, endTS):
    if endTS <= startTS:
        raise ValueError("endTS must be greater than startTS.")

    tickTest = TickTest()
    classifications = tickTest.classifyAll(trades, startTS, endTS)
    if not classifications:
        return 0.0

    imbalanced_volume = 0.0
    iClass = 0
    for i in range(0, trades.getN()):
        timestamp = trades.getTimestamp(i)
        if timestamp < startTS:
            continue
        if timestamp >= endTS:
            break
        if iClass >= len(classifications):
            break
        size = float(trades.getSize(i))
        if not math.isfinite(size):
            iClass += 1
            continue
        imbalanced_volume += float(classifications[iClass][2]) * size
        iClass += 1

    vwap_value = float(VWAP(trades, startTS, endTS).getVWAP())
    if not math.isfinite(vwap_value):
        return 0.0
    imbalance = vwap_value * imbalanced_volume
    return imbalance
