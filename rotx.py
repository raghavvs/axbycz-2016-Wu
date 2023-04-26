import numpy as np

def rotx(t, deg=False):
    if deg:
        t = np.radians(t)

    ct = np.cos(t)
    st = np.sin(t)
    R = np.array([
        [1,  0,   0],
        [0,  ct, -st],
        [0,  st,  ct]
    ])

    return R