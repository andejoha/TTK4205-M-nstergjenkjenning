import numpy as np



def calculateY(x_training, N_ROWS):
    Y = np.zeros([N_ROWS, len(x_training[0])])
    for i in range(len(Y)):
        Y[i][0] = 1
        Y[i][1:] = x_training[i][1:]
    return Y



def calculateB(x_training, N_ROWS):
    b = np.zeros(N_ROWS)
    for i in range(len(x_training)):
        if x_training[i][0] == 1:
            b[i] = 1
        else:
            b[i] = -1
    return b



def leastSquareMethod(training_set, test_set, N_ROWS, N_COLLUMS, best_error_rate_dimension):
    print("\n# ========== Running: Least square method classifier ======== # ")

    error_rate_storage = []

    for dimension in best_error_rate_dimension:
        # Restructuring data sets
        x_training = np.zeros([N_ROWS, np.count_nonzero(dimension == 1) + 1])
        x_training[:, 0] = training_set[:, 0]
        x_test = np.zeros([N_ROWS, np.count_nonzero(dimension == 1) + 2])
        x_test[:, 0] = training_set[:, 0]
        index = 1
        for i in range(len(dimension)):
            if dimension[i] == 1:
                for j in range(N_ROWS):
                    x_training[j][index] = training_set[j][i + 1]
                    x_test[j][index] = test_set[j][i + 1]
                index += 1


        # Create b
        b = calculateB(x_training, N_ROWS)

        # Create Y
        Y = calculateY(x_training, N_ROWS)

        # a = (Y^T*Y)^-1*Y*b^T
        a = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(Y), Y)), np.transpose(Y)), b)

        for i in range(N_ROWS):
            # g = a^T*y
            g = np.matmul(np.transpose(a), Y[i])

            if g >= 0:
                x_test[i][-1] = 1
            else:
                x_test[i][-1] = 2


        # Esimating error rate
        error = 0
        for i in range(N_ROWS):
            if x_test[i][-1] != x_test[i][0]:
                error += 1
        error_rate = error/N_ROWS
        error_rate_storage.append(error_rate)

        print("Error rate: ", "%.2f" % error_rate, " for dimension: ", dimension)

    return x_test