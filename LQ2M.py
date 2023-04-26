import numpy as np

def LQ2M(q):
    q0, qx, qy, qz = q
    M = np.array(
        [
            [q0, -qx, -qy, -qz],
            [qx, q0, -qz, qy],
            [qy, qz, q0, -qx],
            [qz, -qy, qx, q0],
        ]
    )

    return M