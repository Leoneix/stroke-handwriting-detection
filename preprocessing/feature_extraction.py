import numpy as np


def extract_features(points):

    x = points[:,0]
    y = points[:,1]

    width = x.max() - x.min()
    height = y.max() - y.min()

    dx = np.diff(x)
    dy = np.diff(y)

    length = np.sum(np.sqrt(dx**2 + dy**2))

    angle = np.arctan2(dy, dx)

    mean_angle = np.mean(angle)
    std_angle = np.std(angle)

    features = [
        width,
        height,
        length,
        mean_angle,
        std_angle
    ]

    return np.array(features)