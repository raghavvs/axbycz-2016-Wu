import numpy as np

def rotationTheta(g):
    tr = (np.trace(g[0:3, 0:3]) - 1) / 2

    if tr > 1:
        tr = 1
    elif tr < -1:
        tr = -1

    theta = np.arccos(tr)
    return theta