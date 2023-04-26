import numpy as np

def GetJ(RA_noise, RY_sln, RC_noise):
    # refer to eq. (38)
    # GetJ calculates J matrix for Jt=p.

    M = RA_noise.shape[2]  # number of measurement configurations
    J = np.zeros((3 * M, 9))

    for i in range(M):
        J[3 * i - 2:3 * i, :] = np.hstack((RA_noise[:, :, i], -np.eye(3), -RY_sln @ RC_noise[:, :, i]))

    return J