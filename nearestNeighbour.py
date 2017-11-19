import numpy as np



def nearestNeighbour(training_set, test_set, N_ROWS, dimension_matrix):
    print("\n# ========== Running: Nearest neighbour classifier ========== #")
    min_error_rate = float('inf')

    for dimension in dimension_matrix:
        error = 0

        for test_row in range(N_ROWS):
            min_dinstance = float('inf')

            for training_row in range(N_ROWS):
                distance = abs(np.linalg.norm(test_set[test_row][1:-1]*dimension - training_set[training_row][1:]*dimension))
                if distance < min_dinstance:
                    min_dinstance = distance
                    test_set[test_row][-1] = training_set[training_row][0]


            # Estimating error rate
            if test_set[test_row][-1] != test_set[test_row][0]:
                error += 1
        error_rate = error/N_ROWS

        if error_rate < min_error_rate:
            min_error_rate = error_rate
            min_error_rate_dimension = dimension

        print("Minimum error rate: ", "%.2f" % error_rate, " for dimension: ", dimension)

    return test_set, min_error_rate, min_error_rate_dimension