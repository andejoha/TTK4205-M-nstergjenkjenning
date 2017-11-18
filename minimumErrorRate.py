import numpy as np



def minimumErrorRate(training_set, test_set, N_ROWS, best_error_rate_dimension):
    print("\n# ========== Running: Minimum error rate classifier ========= #")

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
        x_training = [x_training[x_training[:, 0] == 1], x_training[x_training[:, 0] == 2]]


        # Calculate mean
        mu = [x_training[0][:, 1:].mean(axis=0), x_training[1][:, 1:].mean(axis=0)]

        # Calculate covariance
        Sigma = [np.cov(x_training[0][:, 1:], rowvar=False), np.cov(x_training[1][:, 1:], rowvar=False)]

        # W_i = -1/2*Sigma_i^-1     i = 1,...,c
        W = [-np.linalg.inv(np.atleast_2d(Sigma[0]))/2, -np.linalg.inv(np.atleast_2d(Sigma[1]))/2]

        # w = Sigma_i^-1*mu_i   i = 1,...,c
        w = [np.matmul(np.linalg.inv(np.atleast_2d(Sigma[0])), mu[0]),
             np.matmul(np.linalg.inv(np.atleast_2d(Sigma[1])), mu[1])]

        # w_0 = -1/2*mu_i^T*Sigma_i^-1*mu_i - 1/2*ln(|Sigma_i|) + ln(P(omega_i))    i = 1,...,c
        w_0 = [-np.matmul(np.matmul(np.transpose(mu[0]), np.linalg.inv(np.atleast_2d(Sigma[0]))), mu[0]) / 2
               - np.log(np.linalg.det(np.atleast_2d(Sigma[0]))) / 2 + np.log(0.5),
               -np.matmul(np.matmul(np.transpose(mu[1]), np.linalg.inv(np.atleast_2d(Sigma[1]))), mu[1]) / 2
               - np.log(np.linalg.det(np.atleast_2d(Sigma[1]))) / 2 + np.log(0.5)]

        for j in range(N_ROWS):
            # g_i = x^T*W_i*x + w^T*x + w_0_i   i = 1,...,c
            g_1 = np.matmul(np.matmul(np.transpose(x_test[j][1:-1]), W[0]), x_test[j][1:-1]) + \
                  np.matmul(np.transpose(w[0]), x_test[j][1:-1]) + w_0[0]
            g_2 = np.matmul(np.matmul(np.transpose(x_test[j][1:-1]), W[1]), x_test[j][1:-1]) + \
                  np.matmul(np.transpose(w[1]), x_test[j][1:-1]) + w_0[1]

            if (g_1 - g_2) >= 0:
                x_test[j][-1] = 1
            else:
                x_test[j][-1] = 2


        # Esimating error rate
        error = 0
        for i in range(N_ROWS):
            if x_test[i][-1] != x_test[i][0]:
                error += 1
        error_rate = error / N_ROWS
        error_rate_storage.append(error_rate)

        print("Error rate: ", "%.2f" % error_rate, " for dimension: ", dimension)

    return x_test