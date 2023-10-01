import numpy as np


def my_argwhere(cond):
    ret = np.argwhere(cond)
    ret = ret.reshape(len(ret))
    return ret


def dB(value, base):
    ratio = value / base
    ratio = np.where(ratio > 0, ratio, np.abs(np.gradient(ratio)).min())
    dB_ratio = 20.0 * np.log10(ratio)
    return dB_ratio
