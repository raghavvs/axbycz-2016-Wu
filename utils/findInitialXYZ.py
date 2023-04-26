import numpy as np
from scipy.linalg import expm, logm
from utils import RQ2M, R2Q, LQ2M, GetWC, qXYZ2RXYZ, vlogR

def findInitialXYZ(RA, RB, RC):
  """FindInitialXYZ Initial solution of AXB=YCZ

  Args:
    RA, RB, RC:                 Rotation matrix, 3x3xN
    RX0, RY0, RZ0:              Initial rotation matrix, 3x3

  """

  # Get the number of motions.
  N_motion = RA.shape[2]

  # Get the minimum number of data.
  M_data = min(N_motion, 10)

  # Initialize the rotation matrices.
  WAB = np.zeros((4 * M_data, 4))
  WC = np.zeros((4 * M_data, 16))

  # Iterate over the motions.
  for i in range(M_data):
    # Get the rotation matrix for the X component.
    WAB[4 * i - 3:4 * i, 1:4] = RQ2M(R2Q(RB[:, :, i])) * LQ2M(R2Q(RA[:, :, i]))

    # Get the rotation matrix for the Z component.
    WC[4 * i - 3:4 * i, 1:16] = GetWC(R2Q(RC[:, :, i]))

  # Initialize the minimum eigenvalues.
  lamda_min1 = np.inf
  lamda_min2 = np.inf
  lamda_min3 = np.inf
  lamda_min4 = np.inf
  lamda_min5 = np.inf

  # Iterate over all possible combinations of the rotation matrices.
  for j in range(2 ** (M_data - 1)):
    # Initialize the weighted rotation matrix.
    WABC = np.zeros((4 * M_data, 4))

    # Iterate over the motions.
    for k in range(M_data):
      # Set the weight of the rotation matrix for the X component.
      WABC[4 * k - 3:4 * k, 1:4] = WABC[4 * k - 3:4 * k, 1:4] * (bin(j)[k] - '0')

    # Get the eigenvalues of the weighted rotation matrix.
    [Vtemp, Dtemp] = np.linalg.eig(WABC.T @ WABC)

    # Update the minimum eigenvalue.
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

    # Get the initial rotation matrices.
    RX01, RY01, RZ01 = qXYZ2RXYZ(V1)
    RX02, RY02, RZ02 = qXYZ2RXYZ(V2)
    RX03, RY03, RZ03 = qXYZ2RXYZ(V3)
    RX04, RY04, RZ04 = qXYZ2RXYZ(V4)
    RX05, RY05, RZ05 = qXYZ2RXYZ(V5)

    # Get the errors.
    err1 = 0
    err2 = 0
    err3 = 0
    err4 = 0
    err5 = 0

    for i in range(M_data):
        err1 += np.linalg.norm(np.log(RA[:, :, i] @ RX01 @ RB[:, :, i] @ np.linalg.inv(RY01 @ RC[:, :, i] @ RZ01)))
        err2 += np.linalg.norm(np.log(RA[:, :, i] @ RX02 @ RB[:, :, i] @ np.linalg.inv(RY02 @ RC[:, :, i] @ RZ02)))
        err3 += np.linalg.norm(np.log(RA[:, :, i] @ RX03 @ RB[:, :, i] @ np.linalg.inv(RY03 @ RC[:, :, i] @ RZ03)))
        err4 += np.linalg.norm(np.log(RA[:, :, i] @ RX04 @ RB[:, :, i] @ np.linalg.inv(RY04 @ RC[:, :, i] @ RZ04)))
        err5 += np.linalg.norm(np.log(RA[:, :, i] @ RX05 @ RB[:, :, i] @ np.linalg.inv(RY05 @ RC[:, :, i] @ RZ05)))

    # Get the minimum error.
    errmin = min([err1, err2, err3, err4, err5])

    # Get the initial rotation matrix.
    if err1 == errmin:
        RX0 = RX01
    elif err2 == errmin:
        RX0 = RX02
    elif err3 == errmin:
        RX0 = RX03
    elif err4 == errmin:
        RX0 = RX04
    else:
        RX0 = RX05

    return RX0