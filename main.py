import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd


class Quaternion:
    def __init__(self, position=None, quat=None):
        if position is not None:
            self.r = 0.0
            self.i = np.array([position[0], position[1], position[2]])
            self.conj = np.array([self.r, -self.i[0], -self.i[1], -self.i[2]])
        elif quat is not None:
            self.r = quat[0]
            self.i = np.array([quat[1], quat[2], quat[3]])
            self.conj = np.array([self.r, -self.i[0], -self.i[1], -self.i[2]])
        else:
            self.i = np.array([0.0, 0.0, 0.0])
            self.r = 0.0
            self.conj = np.array([0.0, 0.0, 0.0, 0.0])

    def rotatePosition(self, pos):
        tmp = self.multiply(pos)

        result = tmp.multiply(Quaternion(quat=self.conj))

        return [result.i[0], result.i[1], result.i[2]]

    def multiply(self, other):
        tmp = Quaternion()
        tmp.r = (self.r * other.r) - (self.i[0] * other.i[0]) - (self.i[1] * other.i[1]) - (self.i[2] * other.i[2])
        tmp.i[0] = (self.r * other.i[0]) + (self.i[0] * other.r) + (self.i[1] * other.i[2]) - (self.i[2] * other.i[1])
        tmp.i[1] = (self.r * other.i[1]) - (self.i[0] * other.i[2]) + (self.i[1] * other.r) + (self.i[2] * other.i[0])
        tmp.i[2] = (self.r * other.i[2]) + (self.i[0] * other.i[1]) - (self.i[1] * other.i[0]) + (self.i[2] * other.r)
        return tmp


def main():
    distanceMatrix = np.loadtxt('data/internal_dists.txt')
    numRows, numCols = distanceMatrix.shape
    M = np.ndarray(distanceMatrix.shape)

    # Calc M using distance matrix
    for i in range(numRows):
        for j in range(numCols):
            M[i, j] = (distanceMatrix[0, j] ** 2 + distanceMatrix[i, 0] ** 2 - distanceMatrix[i, j] ** 2) / 2.

    # SVD Decomposition
    U, S, Ut = svd(M)

    for i in range(numRows):
        S[i] = math.sqrt(S[i])

    X = U * S

    # Init coordinates array N * 3
    coordinates = np.ndarray((numRows, 3))

    # Remove noise
    for i in range(numRows):
        for j in range(numCols):
            if 0.01 > X[i][j] > -0.01:
                X[i][j] = 0.0
        coordinates[i] = X[i, :3]

    coordinatesFromFile = getCoordinatesPdbFile('data/1lyd.pdb')

    translateCoordinatesToOrigin(coordinates)
    translateCoordinatesToOrigin(coordinatesFromFile)

    N = getNFromCoordinates(coordinates, coordinatesFromFile)

    eigValuesN, eigVectorsN = np.linalg.eig(N)

    # Find max eigenvalue index
    maxEigValue = -math.inf
    maxEigIndex = 0
    for i in range(len(eigValuesN)):
        if eigValuesN[i] > maxEigValue:
            maxEigValue = eigValuesN[i]
            maxEigIndex = i

    # Init quat form of max eigen vector
    maxEigQuat = Quaternion(quat=eigVectorsN[maxEigIndex])

    # Init rotated coordinates array N * 3
    rotatedCoordinatesFromFile = np.ndarray(coordinatesFromFile.shape)

    # Rotate coordinates
    for i in range(rotatedCoordinatesFromFile.shape[0]):
        rotatedCoordinatesFromFile[i] = maxEigQuat.rotatePosition(Quaternion(position=coordinatesFromFile[i]))

    plotCoordinates(coordinates, rotatedCoordinatesFromFile)

    # Save coordinates for assignment
    # np.savetxt('results/computed_coords.csv', coordinates, delimiter=',')


def getNFromCoordinates(coordinates, coordinatesFromFile):
    sXX = 0
    sXY = 0
    sXZ = 0
    sYX = 0
    sYY = 0
    sYZ = 0
    sZX = 0
    sZY = 0
    sZZ = 0
    size = coordinates.shape[0]
    for i in range(size):
        sXX += (coordinates[i, 0] * coordinatesFromFile[i, 0])
        sXY += (coordinates[i, 0] * coordinatesFromFile[i, 1])
        sXZ += (coordinates[i, 0] * coordinatesFromFile[i, 2])
        sYX += (coordinates[i, 1] * coordinatesFromFile[i, 0])
        sYY += (coordinates[i, 1] * coordinatesFromFile[i, 1])
        sYZ += (coordinates[i, 1] * coordinatesFromFile[i, 2])
        sZX += (coordinates[i, 2] * coordinatesFromFile[i, 0])
        sZY += (coordinates[i, 2] * coordinatesFromFile[i, 1])
        sZZ += (coordinates[i, 2] * coordinatesFromFile[i, 2])

    return np.array([
        [
            sXX + sYY + sZZ, sYZ - sZY, sZX - sXZ, sXY - sYX
        ],
        [
            sYZ - sZY, sXX - sYY - sZZ, sXY + sYX, sZX + sXZ
        ],
        [
            sZX - sXZ, sXY + sYX, -sXX + sYY - sZZ, sYZ + sZY
        ],
        [
            sXY - sYX, sZX + sXZ, sYZ + sZY, -sXX - sYY + sZZ
        ]
    ], np.float)


def translateCoordinatesToOrigin(coordinates):
    sumX = 0.
    sumY = 0.
    sumZ = 0.
    size = coordinates.shape[0]

    # Get average position of both coordinate systems
    for i in range(size):
        sumX += coordinates[i, 0]
        sumY += coordinates[i, 1]
        sumZ += coordinates[i, 2]
    center = np.array([sumX / size, sumY / size, sumZ / size])

    # Subtract vector from each coordinate
    for i in range(size):
        coordinates[i] -= center


def plotCoordinates(coordinates, coordinatesFromFile):
    x, y, z = coordinates.T
    x2, y2, z2 = coordinatesFromFile.T

    ax = plt.axes(projection='3d')

    ax.scatter3D(x, y, z)
    ax.plot3D(x, y, z)
    ax.scatter3D(x2, y2, z2)
    ax.plot3D(x2, y2, z2)
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
