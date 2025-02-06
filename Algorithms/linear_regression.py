# Linear Regression

# Write a python function manually that represents Linear Regression. Do not use 3rd party libraries.

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3X + noise

# Initialize parameters
m = 0  # Slope
b = 0  # Intercept
alpha = 0.1  # Learning rate
epochs = 1000  # Number of iterations
n = len(X)  # Number of samples

# Gradient Descent
for _ in range(epochs):
    y_pred = m * X + b
    error = y_pred - y

    m_gradient = (2/n) * np.sum(error * X) 
    b_gradient = (2/n) * np.sum(error)
    m -= alpha * m_gradient
    b -= alpha * b_gradient

# Final Parameters
print(f"Learned Parameters: m = {m:.2f}, b = {b:.2f}")

# Plot results
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, m * X + b, color="red", label="Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
