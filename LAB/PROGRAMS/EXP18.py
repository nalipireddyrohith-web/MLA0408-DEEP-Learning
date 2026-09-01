import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier

# Generate circular dataset
X, y = make_circles(
    n_samples=300,
    noise=0.1,
    factor=0.5
)

# Create MLP neural network model
model = MLPClassifier(
    hidden_layer_sizes=(2, 2),
    activation='tanh',
    solver='sgd',
    learning_rate_init=0.1,
    max_iter=1000,
    random_state=1
)

# Train the model
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate accuracy
accuracy = model.score(X, y)

# Display model details
print("Learning Rate:", 0.1)
print("Activation:", "Tanh")
print("Hidden Layers:", 2)
print("Hidden Neurons:", 2)
print("Accuracy:", accuracy)

# Plot the circular dataset
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Neural Network Analysis of Circular Data")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
