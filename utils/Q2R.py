import numpy as np

def Q2R(Q):
  """Convert a quaternion to a rotation matrix.

  Args:
    Q: A 3-element array of quaternions.

  Returns:
    A 3x3 rotation matrix.
  """

  # Check if the Q array has three values.
  if len(Q) != 3:
    raise ValueError("Q must have 3 values.")

  # Unpack the Q array into three values.
  q0, q1, q2 = Q

  # Create a rotation matrix from the quaternion.
  R = np.array([
      [q0 ** 2 + q1 ** 2 - q2 ** 2, 2 * q1 * q2, 2 * q1 * q3],
      [2 * q1 * q2, q0 ** 2 - q1 ** 2 + q2 ** 2, 2 * q2 * q3],
      [2 * q1 * q3, 2 * q2 * q3, q0 ** 2 - q1 ** 2 - q2 ** 2],
  ])

  return R