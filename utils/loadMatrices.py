import numpy as np

def loadMatrices(filepath):
    matrices = []
    matrix = np.zeros((4, 4))
    line_counter = 0

    with open(filepath, 'r') as file:
        for line in file:
            if line_counter == 4:
                matrices.append(matrix)
                matrix = np.zeros((4, 4))
                line_counter = 0

            values = line.strip().split()
            for j, value in enumerate(values):
                matrix[line_counter, j] = float(value)

            line_counter += 1

    # Append the last matrix if necessary
    if line_counter == 4:
        matrices.append(matrix)

    if len(matrices) == 0:
        raise ValueError("No matrices were loaded. Check your file and try again.")
    else:
        return np.stack(matrices, axis=2)  # Stack matrices along the third axis