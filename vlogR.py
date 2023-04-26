import numpy as np
import warnings
import skew

def vlogR(R):
    DELTA = 1e-12

    tr = (np.trace(R) - 1) / 2

    if tr > 1:
        tr = 1
    elif tr < -1:
        tr = -1

    fai = np.arccos(tr)

    if abs(fai) < DELTA:
        w = np.array([0, 0, 0])

    elif abs(fai - np.pi) < DELTA:
        warnings.warn('Logarithm of rotation matrix with angle PI.')

        eig_val, eig_vec = np.linalg.eig(R)

        eig_val = np.real(eig_val)
        eig_vec = np.real(eig_vec)

        max_idx = np.argmax(eig_val)

        w = eig_vec[:, max_idx]

        if np.max(np.max(np.cos(fai) * np.eye(3) + (1 - np.cos(fai)) * np.outer(w, w) + np.sin(fai) * skew(w) - R)) > DELTA:
            w = -w

        w = w * fai

    else:
        w = fai / (2 * np.sin(fai)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    return w