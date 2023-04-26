import numpy as np
from utils import skew

def rotationMatrix(w, theta):
    R = np.eye(3) + skew(w) * np.sin(theta) + np.dot(skew(w), skew(w)) * (1 - np.cos(theta))
    return R