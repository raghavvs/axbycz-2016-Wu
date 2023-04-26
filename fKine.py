import numpy as np

def fKine(q):
    d1 = 0.290
    a2 = 0.270
    a3 = 0.070
    d4 = 0.302
    d5 = 0.072
    d6 = 0.074

    theta = q

    d = np.array([d1, 0, 0, d4, d5, d6])
    a = np.array([0, a2, a3, 0, 0, 0])
    alpha = np.array([-np.pi/2, 0, -np.pi/2, np.pi/2, -np.pi/2, 0])

    T = np.eye(4)
    for i in range(6):
        Ti = np.array([
            [np.cos(theta[i]), -np.sin(theta[i]) * np.cos(alpha[i]), np.sin(theta[i]) * np.sin(alpha[i]), a[i] * np.cos(theta[i])],
            [np.sin(theta[i]), np.cos(theta[i]) * np.cos(alpha[i]), -np.cos(theta[i]) * np.sin(alpha[i]), a[i] * np.sin(theta[i])],
            [0, np.sin(alpha[i]), np.cos(alpha[i]), d[i]],
            [0, 0, 0, 1]
        ])
        T = np.dot(T, Ti)
    return T