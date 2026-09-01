import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Multi-Class Dataset (Iris Dataset)
data = load_iris()
X = data.data
y = data.target

# Split the dataset into Training and Testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create Neural Network Model
model = MLPClassifier(
    hidden_layer_sizes=(2, 2),  # 2 Hidden Layers, 2 Neurons each
    activation='tanh',          # Tanh Activation Function
    learning_rate_init=0.1,    # Learning Rate
    max_iter=1000,
    random_state=42
)

# Train the Model
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display Results
print("Neural Network Analysis for Multi-Class Data")
print("Learning Rate: 0.1")
print("Activation Function: Tanh")
print("Hidden Layers: 2")
print("Hidden Neurons: 2")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
