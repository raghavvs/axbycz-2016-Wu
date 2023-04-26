import numpy as np
import os

def loadMatrices(filepaths):
    matrices = []
    for filepath in filepaths:
        matrix = np.zeros((4, 4))
        try:
            with open(filepath, 'r') as file:
                for i in range(4):
                    for j in range(4):
                        matrix[i, j] = float(file.readline().strip())
                matrices.append(matrix)
        except IOError:
            print(f"Unable to open the file: {filepath}")
    return matrices