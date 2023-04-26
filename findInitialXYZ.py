import numpy as np
from scipy.linalg import expm, logm
import RQ2M, R2Q, LQ2M, GetWC, qXYZ2RXYZ, vlogR  # Import the necessary functions

def FindInitialXYZ(RA, RB, RC):
    N_motion = RA.shape[2]

    M_data = min(N_motion, 10)
    WAB = np.zeros((4 * M_data, 4))
    WC = np.zeros((4 * M_data, 16))

    for i in range(M_data):
        WAB[4 * i:4 * (i + 1), :] = np.dot(RQ2M(R2Q(RB[:, :, i])), LQ2M(R2Q(RA[:, :, i])))
        WC[4 * i:4 * (i + 1), :] = GetWC(R2Q(RC[:, :, i]))

    lamda_min1 = np.inf
    lamda_min2 = np.inf
    lamda_min3 = np.inf
    lamda_min4 = np.inf
    lamda_min5 = np.inf
    V1 = V2 = V3 = V4 = V5 = None

    for j in range(2 ** (M_data - 1)):
        WABC = np.hstack((WAB, -WC))
        for k in range(M_data):
            WABC[4 * k:4 * (k + 1), :] *= 2 * ((j >> k) & 1) - 1

        eig_values, Vtemp = np.linalg.eig(np.dot(WABC.T, WABC))
        Dtemp = np.diag(eig_values)

        if Dtemp[0, 0] < lamda_min1:
            lamda_min5 = lamda_min4
            V5 = V4

            lamda_min4 = lamda_min3
            V4 = V3

            lamda_min3 = lamda_min2
            V3 = V2

            lamda_min2 = lamda_min1
            V2 = V1

            lamda_min1 = Dtemp[0, 0]
            V1 = Vtemp

        elif Dtemp[0, 0] < lamda_min2:
            lamda_min5 = lamda_min4
            V5 = V4

            lamda_min4 = lamda_min3
            V4 = V3

            lamda_min3 = lamda_min2
            V3 = V2

            lamda_min2 = Dtemp[0, 0]
            V2 = Vtemp

        elif Dtemp[0, 0] < lamda_min3:
            lamda_min5 = lamda_min4
            V5 = V4

            lamda_min4 = lamda_min3
            V4 = V3

            lamda_min3 = Dtemp[0, 0]
            V3 = Vtemp

        elif Dtemp[0, 0] < lamda_min4:
            lamda_min5 = lamda_min4
            V5 = V4

            lamda_min4 = Dtemp[0, 0]
            V4 = Vtemp

        elif Dtemp[0, 0] < lamda_min5:
            lamda_min5 = Dtemp[0, 0]
            V5 = Vtemp

    RX01, RY01, RZ01 = qXYZ2RXYZ(V1)
    RX02, RY02, RZ02 = qXYZ2RXYZ(V2)
    RX03, RY03, RZ03 = qXYZ2RXYZ(V3)
    RX04, RY04, RZ04 = qXYZ2RXYZ(V4)
    RX05, RY05, RZ05 = qXYZ2RXYZ(V5)

    err1 = err2 = err3 = err4 = err5 = 0
    for i in range(M_data):
        err1 += np.linalg.norm(vlogR(np.dot(RA[:, :, i], np.dot(RX01, np.dot(RB[:, :, i], RY01.T @ RC[:, :, i] @ RZ01.T)))))
        err2 += np.linalg.norm(vlogR(np.dot(RA[:, :, i], np.dot(RX02, np.dot(RB[:, :, i], RY02.T @ RC[:, :, i] @ RZ02.T)))))
        err3 += np.linalg.norm(vlogR(np.dot(RA[:, :, i], np.dot(RX03, np.dot(RB[:, :, i], RY03.T @ RC[:, :, i] @ RZ03.T)))))
        err4 += np.linalg.norm(vlogR(np.dot(RA[:, :, i], np.dot(RX04, np.dot(RB[:, :, i], RY04.T @ RC[:, :, i] @ RZ04.T)))))
        err5 += np.linalg.norm(vlogR(np.dot(RA[:, :, i], np.dot(RX05, np.dot(RB[:, :, i], RY05.T @ RC[:, :, i] @ RZ05.T)))))

    errmin = min([err1, err2, err3, err4, err5])
    if err1 == errmin:
        RX0, RY0, RZ0 = RX01, RY01, RZ01
    elif err2 == errmin:
        RX0, RY0, RZ0 = RX02, RY02, RZ02
    elif err3 == errmin:
        RX0, RY0, RZ0 = RX03, RY03, RZ03
    elif err4 == errmin:
        RX0, RY0, RZ0 = RX04, RY04, RZ04
    elif err5 == errmin:
        RX0, RY0, RZ0 = RX05, RY05, RZ05

    return RX0, RY0, RZ0