# Logistic Regression

# Write a python function manually that represents Logistic Regression. Do not use 3rd party libraries.

import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic Regression using Gradient Descent
def logistic_regression(X, y, alpha=0.1, epochs=1000):
    n = len(X)  # Number of samples
    m, b = 0, 0  # Initialize parameters
    
    # Gradient Descent
    for _ in range(epochs):
        z = m * X + b  # Compute linear function
        y_pred = sigmoid(z)  # Apply sigmoid
        
        # Compute gradients
        m_gradient = (1/n) * np.sum((y_pred - y) * X)
        b_gradient = (1/n) * np.sum(y_pred - y)
        
        # Update parameters
        m -= alpha * m_gradient
        b -= alpha * b_gradient

    return m, b

# Generate synthetic data (binary classification problem)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # Features
y = (X > 1).astype(int)  # Labels: 1 if X > 1, else 0

# Train the model
m, b = logistic_regression(X, y)

# Plot decision boundary
X_boundary = np.linspace(0, 2, 100)
y_boundary = sigmoid(m * X_boundary + b)

plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X_boundary, y_boundary, color="red", label="Sigmoid Curve")
plt.xlabel("X")
plt.ylabel("Probability")
plt.legend()
plt.show()

print(f"Learned Parameters: m = {m:.2f}, b = {b:.2f}")
