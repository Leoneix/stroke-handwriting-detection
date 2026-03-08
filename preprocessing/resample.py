import numpy as np

def resample(points, n=64):

    idx = np.linspace(0, len(points)-1, n)

    idx = idx.astype(int)

    return points[idx]
