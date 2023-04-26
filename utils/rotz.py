import numpy as np

def rotz(t, deg=False):
    if deg:
        t = np.radians(t)

    ct = np.cos(t)
    st = np.sin(t)
    R = np.array([
        [ct, -st, 0],
        [st,  ct, 0],
        [0,   0,  1]
    ])

    return R