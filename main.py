import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd


def main():
    distanceMatrix = np.loadtxt('data/internal_dists.txt')
    numRows, numCols = distanceMatrix.shape
    newMatrix = np.ndarray(distanceMatrix.shape)
    for i in range(numRows):
        for j in range(numCols):
            newMatrix[i, j] = (distanceMatrix[0, j]**2 + distanceMatrix[i, 0]**2 - distanceMatrix[i, j]**2) / 2.

    # print(newMatrix)
    U, S, Ut = svd(newMatrix)

    for i in range(numRows):
        S[i] = math.sqrt(S[i])

    X = U * S
    coordinates = np.ndarray((numRows, 3))
    for i in range(numRows):
        for j in range(numCols):
            if 0.01 > X[i][j] > -0.01:
                X[i][j] = 0.0
        coordinates[i] = X[i, :3]

    coordinatesFromFile = getCoordinatesPdbFile('data/1lyd.pdb')

    x, y, z = coordinates.T
    np.savetxt('results/computed_coords.csv', coordinates, delimiter=',')
    x2, y2, z2 = coordinatesFromFile.T

    ax = plt.axes(projection='3d')

    ax.scatter3D(x, y, z)
    ax.plot3D(x, y, z, 'green')
    ax.scatter3D(x2, y2, z2)
    ax.plot3D(x2, y2, z2, 'green')
    plt.show()


def getCoordinatesPdbFile(fileName):
    coordinateList = np.ndarray((164, 3))
    i = 0
    for line in open(fileName):
        line_ = line.split()
        try:
            type_ = line_[2]
        except IndexError:
            type_ = 'null'
        if type_ == 'CA':
            coordinateList[i] = (np.array([float(line_[6]), float(line_[7]), float(line_[8])]))
            i += 1

    return coordinateList


if __name__ == '__main__':
    main()
