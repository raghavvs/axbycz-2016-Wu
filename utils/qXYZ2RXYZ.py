import numpy as np
from utils import Q2R

def qXYZ2RXYZ(V):
  """Obtain RX0, RY0, RZ0 from V.

  Args:
    V: A 4x4 array of quaternions.

  Returns:
    RX0, RY0, RZ0: A 3x3 array of rotation matrices.
  """

  # Get the X component of the quaternion.
  qX = V[:, 0] / np.linalg.norm(V[:, 0])

  # Get the Y and Z components of the quaternion.
  qYZ = V[:, 1:4] / np.linalg.norm(V[:, 0])

  # Get the rotation matrix for the X component.
  RX0 = Q2R(qX)

  # Make the sign of qY0 * qZ0 plus.
  qYZ *= np.sign(qYZ[0])

  # Get the signs of qY0 and qZ0.
  SignY0 = 1
  SignY1 = np.sign(qYZ[1])
  SignY2 = np.sign(qYZ[2])
  SignY3 = np.sign(qYZ[3])

  # Get the Y components of the quaternion.
  qY0 = np.linalg.norm(qYZ[:, 0]) * SignY0
  qY1 = np.linalg.norm(qYZ[:, 1]) * SignY1
  qY2 = np.linalg.norm(qYZ[:, 2]) * SignY2
  qY3 = np.linalg.norm(qYZ[:, 3]) * SignY3

  # Get the rotation matrix for the Y component.
  qY = np.array([qY0, qY1, qY2, qY3]) / np.linalg.norm(qY)

  # Get the signs of qZ0.
  SignZ0 = 1
  SignZ1 = np.sign(qYZ[4])
  SignZ2 = np.sign(qYZ[5])
  SignZ3 = np.sign(qYZ[6])

  # Get the Z components of the quaternion.
  qZ0 = np.linalg.norm(qYZ[:, 0]) * SignZ0
  qZ1 = np.linalg.norm(qYZ[:, 1]) * SignZ1
  qZ2 = np.linalg.norm(qYZ[:, 2]) * SignZ2
  qZ3 = np.linalg.norm(qYZ[:, 3]) * SignZ3

  # Get the rotation matrix for the Z component.
  qZ = np.array([qZ0, qZ1, qZ2, qZ3]) / np.linalg.norm(qZ)

  # Return the rotation matrices.
  return RX0, qY, qZ