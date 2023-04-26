import numpy as np

def R2Q(R):
    Qxx, Qyy, Qzz = np.diag(R)
    Qzy, Qyz, Qxz, Qzx, Qyx, Qxy = R[2, 1], R[1, 2], R[0, 2], R[2, 0], R[1, 0], R[0, 1]

    t = Qxx + Qyy + Qzz

    if t >= 0:
        r = np.sqrt(1 + t)
        s = 0.5 / r
        w = 0.5 * r
        x = (Qzy - Qyz) * s
        y = (Qxz - Qzx) * s
        z = (Qyx - Qxy) * s
    elif abs(Qxx) >= max(abs(Qyy), abs(Qzz)):
        r = np.sqrt(1 + Qxx - Qyy - Qzz)
        s = 0.5 / r
        w = (Qzy - Qyz) * s
        x = 0.5 * r
        y = (Qxy + Qyx) * s
        z = (Qzx + Qxz) * s
    elif abs(Qyy) >= max(abs(Qxx), abs(Qzz)):
        r = np.sqrt(1 - Qxx + Qyy - Qzz)
        s = 0.5 / r
        w = (Qxz - Qzx) * s
        x = (Qxy + Qyx) * s
        y = 0.5 * r
        z = (Qyz + Qzy) * s
    else:
        r = np.sqrt(1 - Qxx - Qyy + Qzz)
        s = 0.5 / r
        w = (Qyx - Qxy) * s
        x = (Qxz + Qzx) * s
        y = (Qyz + Qzy) * s
        z = 0.5 * r

    Q = np.array([w, x, y, z])

    return Q