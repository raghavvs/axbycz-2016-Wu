import numpy as np

def RQ2M(q):
  """Right Multiplier Matrix from a quaternion

  Args:
    q: Quaternion, (4,)

  Returns:
    Rotation matrix, (4, 4)
  """

  # Check the shape of the quaternion.
  if q.shape != (4,):
    raise ValueError("The shape of the quaternion must be (4,).")

  # Convert the quaternion to a rotation matrix.
  q0 = q[0]
  qx = q[1]
  qy = q[2]
  qz = q[3]

  M = np.array([
      [q0, -qx, -qy, -qz],
      [qx,  q0,  qz,  -qy],
      [qy,  -qz, q0,  qx],
      [qz,  qy,  -qx, q0]
  ])

  return M