# Neural Network Analysis for Two-Class Data
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
# Create Two-Class Dataset
X, y = make_classification(
n_samples=500,
n_features=4,
n_classes=2,
n_clusters_per_class=1,
random_state=42
)
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
hidden_layer_sizes=(2, 2, 2), # 3 Hidden Layers, 2 Neurons each
activation='relu', # ReLU Activation Function
learning_rate_init=0.001, # Learning Rate
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
print("Neural Network Analysis for Two-Class Data")
print("-------------------------------------------")
print("Learning Rate: 0.001")
print("Activation Function: ReLU")
print("Hidden Layers: 3")
print("Hidden Neurons: 2")
print("-------------------------------------------")
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
