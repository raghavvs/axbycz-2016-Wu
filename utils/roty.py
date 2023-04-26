import numpy as np

def roty(t, deg=False):
    if deg:
        t = np.radians(t)

    ct = np.cos(t)
    st = np.sin(t)
    R = np.array([
        [ct,  0,  st],
        [0,   1,  0],
        [-st, 0,  ct]
    ])

    return R