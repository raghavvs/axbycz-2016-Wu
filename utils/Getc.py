import numpy as np

def Getc(RA_noise, RB_noise, RC_noise, RX_init, RY_init, RZ_init):
    M = RA_noise.shape[2]
    c = np.zeros((9 * M, 1))

    for i in range(M):
        RA = RA_noise[:, :, i]
        RB = RB_noise[:, :, i]
        RC = RC_noise[:, :, i]
        RAXBYCZ = -RA @ RX_init @ RB + RY_init @ RC @ RZ_init
        c[9 * i : 9 * (i + 1)] = RAXBYCZ.reshape((9, 1))

    return c