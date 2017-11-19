import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from nearestNeighbour import nearestNeighbour
from minimumErrorRate import minimumErrorRate
from leastSquareMethod import leastSquareMethod



def readData(file_name):
    file = open(file_name, 'r')
    lines = file.readlines()

    N_ROWS = len(lines)//2
    N_COLLUMS = len(lines[0].split())

    training_set = np.zeros([N_ROWS, N_COLLUMS])
    test_set = np.zeros([N_ROWS, N_COLLUMS + 1])

    for i in range(N_ROWS):
        for j in range(N_COLLUMS):
            training_set[i][j] = float(lines[i*2].split()[j])
            test_set[i][j] = float(lines[i*2 + 1].split()[j])

    return N_ROWS, N_COLLUMS, training_set, test_set



# Returns a matrix with all possible dimension combinations
def constructDimentionMatrix(dim, N_DIM):
    for i in range(N_DIM):
        combinations = list(itertools.combinations(list(range(N_DIM)), dim + 1))

        dimension_matrix = np.zeros([len(combinations), N_DIM])

        row = 0
        for indexes in combinations:
            for col in indexes:
                dimension_matrix[row][col] = 1
            row += 1

        return dimension_matrix



def plotResults(best_error_rate_dimension, training_set, test_set, N_ROWS, N_COLLUMS, data_file):
    data_file,_ = data_file.split(".")
    name = "results/" + data_file


    dimension = [best_error_rate_dimension[1]]

    result_NN,_,_ = nearestNeighbour(training_set, test_set, N_ROWS, dimension)

    x_test = np.zeros([N_ROWS, np.count_nonzero(dimension[0] == 1) + 2])
    x_test[:, 0] = result_NN[:, 0]
    x_test[:, -1] = result_NN[:, -1]
    index = 1
    for i in range(len(dimension[0])):
        if dimension[0][i] == 1:
            for j in range(N_ROWS):
                x_test[j][index] = result_NN[j][i + 1]
            index += 1

    result_MER = minimumErrorRate(training_set, test_set, N_ROWS, dimension)
    result_LSM = leastSquareMethod(training_set, test_set, N_ROWS, N_COLLUMS, dimension)


    plt.figure(1)
    for row in x_test:
        if row[0] == 1:
            plt.plot(row[1], row[2], 'ro')
        else:
            plt.plot(row[1], row[2], 'bo')
    plt.legend(handles=[mpatches.Patch(color='red', label='Class 1'), mpatches.Patch(color='blue', label='Class 2')])
    t1 = "Original plot before classifying for dimension: "
    t2 = ''.join(str(i) for i in dimension)
    plt.title(t1 + t2)
    plt.savefig(name + " Original plot.png")

    plt.figure(2)
    for row in x_test:
        if row[-1] == 1:
            plt.plot(row[1],row[2], 'ro')
        else:
            plt.plot(row[1],row[2], 'bo')
    plt.legend(handles=[mpatches.Patch(color='red', label='Class 1'), mpatches.Patch(color='blue', label='Class 2')])
    t1 = "Nearest neighbour classifier for dimension: "
    t2 = ''.join(str(i) for i in dimension)
    plt.title(t1 + t2)
    plt.savefig(name + " NN plot.png")

    plt.figure(3)
    for row in result_MER:
        if row[-1] == 1:
            plt.plot(row[1],row[2], 'ro')
        else:
            plt.plot(row[1],row[2], 'bo')
    plt.legend(handles=[mpatches.Patch(color='red', label='Class 1'), mpatches.Patch(color='blue', label='Class 2')])
    t1 = "Minimum error rate classifier for dimension: "
    t2 = ''.join(str(i) for i in dimension)
    plt.title(t1 + t2)
    plt.savefig(name + " MER plot.png")

    plt.figure(4)
    for row in result_LSM:
        if row[-1] == 1:
            plt.plot(row[1],row[2], 'ro')
        else:
            plt.plot(row[1],row[2], 'bo')
    plt.legend(handles=[mpatches.Patch(color='red', label='Class 1'), mpatches.Patch(color='blue', label='Class 2')])
    t1 = "Least Square method classifier for dimension: "
    t2 = ''.join(str(i) for i in dimension)
    plt.title(t1 + t2)
    plt.savefig(name + " LSM plot.png")

    plt.show()



if __name__ == "__main__":
    data_file = "ds-1.txt"
    #data_file = "ds-2.txt"
    #data_file = "ds-3.txt"
    N_ROWS, N_COLLUMS, training_set, test_set = readData(data_file)

    best_error_rate_dimension = []
    for current_dimension in range(N_COLLUMS - 1):
        dimension_matrix = constructDimentionMatrix(current_dimension, N_COLLUMS - 1)
        result_NN, min_error_rate, min_error_rate_dimension = nearestNeighbour(training_set, test_set, N_ROWS,
                                                                               dimension_matrix)
        best_error_rate_dimension.append(min_error_rate_dimension)
        print("Best error rate: ", "%.2f" % min_error_rate, " for dimension: ", min_error_rate_dimension)

    minimumErrorRate(training_set, test_set, N_ROWS, best_error_rate_dimension)
    leastSquareMethod(training_set, test_set, N_ROWS, N_COLLUMS, best_error_rate_dimension)

    plotResults(best_error_rate_dimension, training_set, test_set, N_ROWS, N_COLLUMS, data_file)
