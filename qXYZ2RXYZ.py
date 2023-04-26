import numpy as np
import Q2R


def qXYZ2RXYZ(V):
    qX = V[0:4, 0] / np.linalg.norm(V[0:4, 0])

    qYZ = V[4:20, 0] / np.linalg.norm(V[0:4, 0])

    RX0 = Q2R(qX)

    qYZ = qYZ * np.sign(qYZ[0])  # make the sign of qY0*qZ0 plus

    SignY0 = 1
    SignY1 = np.sign(qYZ[4])
    SignY2 = np.sign(qYZ[8])
    SignY3 = np.sign(qYZ[12])
    qY0 = np.linalg.norm(qYZ[0:4]) * SignY0
    qY1 = np.linalg.norm(qYZ[4:8]) * SignY1
    qY2 = np.linalg.norm(qYZ[8:12]) * SignY2
    qY3 = np.linalg.norm(qYZ[12:16]) * SignY3
    qY = np.array([qY0, qY1, qY2, qY3])
    qY = qY / np.linalg.norm(qY)

    SignZ0 = 1
    SignZ1 = np.sign(qYZ[1])
    SignZ2 = np.sign(qYZ[2])
    SignZ3 = np.sign(qYZ[3])
    qZ0 = np.linalg.norm(qYZ[np.array([0, 4, 8, 12])]) * SignZ0
    qZ1 = np.linalg.norm(qYZ[np.array([1, 5, 9, 13])]) * SignZ1
    qZ2 = np.linalg.norm(qYZ[np.array([2, 6, 10, 14])]) * SignZ2
    qZ3 = np.linalg.norm(qYZ[np.array([3, 7, 11, 15])]) * SignZ3
    qZ = np.array([qZ0, qZ1, qZ2, qZ3])
    qZ = qZ / np.linalg.norm(qZ)

    RY0 = Q2R(qY)

    RZ0 = Q2R(qZ)
    return RX0, RY0, RZ0