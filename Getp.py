import numpy as np

def Getp(RA_noise, RX_sln, RY_sln, tA_noise, tB_noise, tC_noise):
    M = RA_noise.shape[2]
    p = np.zeros((3 * M, 1))

    for i in range(M):
        p[3 * i : 3 * (i + 1), 0] = (
            -tA_noise[:, i]
            - RA_noise[:, :, i] @ RX_sln @ tB_noise[:, i]
            + RY_sln @ tC_noise[:, i]
        )

    return p