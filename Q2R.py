import numpy as np

def Q2R(Q):
    q0, q1, q2, q3 = Q
    R = np.array(
        [
            [q0*q0+q1*q1-q2*q2-q3*q3, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2],
            [2*q1*q2+2*q0*q3, q0*q0-q1*q1+q2*q2-q3*q3, 2*q2*q3-2*q0*q1],
            [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, q0*q0-q1*q1-q2*q2+q3*q3],
        ]
    )

    return R