import time

import numpy as np

# Set the random seed
np.random.seed(50)

# Example data for comparison (5 features, 100 examples)
X = np.random.rand(5, 100)  # 5 features, 100 examples
y = np.random.randint(0, 2, (1, 100))  # Random binary labels (0 or 1)

n_x, m = X.shape  # Number of features and number of examples
learning_rate = 0.01
iterations = 1000


### Non-vectorized logistic regression ###
def non_vectorized_logistic_regression(X, y, n_x, m, learning_rate, iterations):
    w = np.zeros((n_x, 1))  # Initialize weights
    b = 0  # Initialize bias

    for iter in range(iterations):
        J = 0  # Cost
        dw = np.zeros((n_x, 1))  # Gradient for weights
        db = 0  # Gradient for bias

        for i in range(m):
            # Forward propagation for each example
            z_i = w.T @ X[:, i] + b  # Linear function
            a_i = 1 / (1 + np.exp(-z_i))  # Sigmoid function

            # Cost function for this example
            J += -(y[0, i] * np.log(a_i) + (1 - y[0, i]) * np.log(1 - a_i))

            # Backward propagation (gradients)
            dz_i = a_i - y[0, i]  # Scalar gradient
            dw += X[:, i].reshape(-1, 1) * dz_i  # Gradient for weights
            db += dz_i  # Gradient for bias

        # Average the cost and gradients
        J = J / m  # J
        dw = dw / m  # dw
        db = db / m  # db

        # Update weights and bias
        w = w - learning_rate * dw  # w
        b = b - learning_rate * db  # b

    return w, b, J


### Vectorized logistic regression ###
def vectorized_logistic_regression(X, y, n_x, m, learning_rate, iterations):
    w = np.zeros((n_x, 1))  # Initialize weights
    b = 0  # Initialize bias

    for iter in range(iterations):
        # Forward propagation
        Z = np.dot(w.T, X) + b  # Linear function (vectorized)
        A = 1 / (1 + np.exp(-Z))  # Sigmoid function (vectorized)

        # Cost function (vectorized)
        J = -(1 / m) * np.sum((y * np.log(A) + (1 - y) * np.log(1 - A)))

        # Backward propagation (vectorized)
        dZ = A - y  # Gradient of cost with respect to Z
        dw = 1 / m * X @ dZ.T  # Gradient with respect to weights
        db = 1 / m * np.sum(dZ)  # Gradient with respect to bias

        # Update weights and bias
        w = w - learning_rate * dw  # w
        b = b - learning_rate * db  # b

    return w, b, J


### Timing and execution ###

# Measure time for non-vectorized version
start_time = time.time()
w_non_vec, b_non_vec, J_non_vec = non_vectorized_logistic_regression(
    X, y, n_x, m, learning_rate, iterations
)
non_vec_time = time.time() - start_time
print(f"Non-vectorized Logistic Regression Time: {non_vec_time:.6f} seconds")

# Measure time for vectorized version
start_time = time.time()
w_vec, b_vec, J_vec = vectorized_logistic_regression(
    X, y, n_x, m, learning_rate, iterations
)
vec_time = time.time() - start_time
print(f"Vectorized Logistic Regression Time: {vec_time:.6f} seconds")

# Compare results
print(f"Cost from non-vectorized: {J_non_vec[0]:.6f}")
print(f"Cost from vectorized: {J_vec:.6f}")
