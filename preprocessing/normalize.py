import numpy as np

def normalize_strokes(sample):

    strokes = sample["strokes"]

    points = []

    for stroke in strokes:
        for p in stroke:
            points.append([p["x"], p["y"]])

    points = np.array(points)

    x = points[:,0]
    y = points[:,1]

    # shift to origin
    x = x - np.mean(x)
    y = y - np.mean(y)

    # scale normalization
    scale = max(np.std(x), np.std(y)) + 1e-6

    x = x / scale
    y = y / scale

    return np.stack([x,y], axis=1)
