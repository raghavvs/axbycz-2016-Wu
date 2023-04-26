import numpy as np
import skew

def GetF(RA_noise, RB_noise, RC_noise, RX_init, RY_init, RZ_init):
    # refer to eq. (29)
    # GetF calculates F matrix for Fr=c.
    
    M = RA_noise.shape[2]  # the number of measurement configurations
    F = np.zeros((9 * M, 9))
    
    for i in range(M):
        RA = RA_noise[:, :, i]
        RB = RB_noise[:, :, i]
        RC = RC_noise[:, :, i]
        RXB = np.dot(RX_init, RB)
        RYCZ = np.dot(np.dot(RY_init, RC), RZ_init)
        
        F[9 * i - 8:9 * i - 5, :] = np.hstack((-RA @ skew(RXB[:, 0]), skew(RYCZ[:, 0]), RY_init @ RC @ skew(RZ_init[:, 0])))
        F[9 * i - 5:9 * i - 2, :] = np.hstack((-RA @ skew(RXB[:, 1]), skew(RYCZ[:, 1]), RY_init @ RC @ skew(RZ_init[:, 1])))
        F[9 * i - 2:9 * i, :] = np.hstack((-RA @ skew(RXB[:, 2]), skew(RYCZ[:, 2]), RY_init @ RC @ skew(RZ_init[:, 2])))
    
    return F