from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
# Load Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create Neural Network
model = MLPClassifier(
hidden_layer_sizes=(2, 2, 2),
activation='relu',
learning_rate_init=0.001,
max_iter=2000,
random_state=42
)
# Train the model
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Neural Network Analysis for Multi-Class Data")
print("Learning Rate: 0.001")
print("Activation Function: ReLU")
print("Hidden Layers: 3")
print("Hidden Neurons: 2")
