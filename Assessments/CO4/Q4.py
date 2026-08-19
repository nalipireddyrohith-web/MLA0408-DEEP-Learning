import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

np.random.seed(42)
tf.random.set_seed(42)

prices = np.array([100,102,104,103,106,108,110,109,112,115,117,116,120,122,125,124,128,130,133,135], dtype=float)

minimum = prices.min()
maximum = prices.max()
scaled = (prices - minimum) / (maximum - minimum)

X = []
y = []
sequence_length = 3

for i in range(len(scaled)-sequence_length):
    X.append(scaled[i:i+sequence_length])
    y.append(scaled[i+sequence_length])

X = np.array(X)
y = np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

model = Sequential([LSTM(32, input_shape=(3,1)), Dense(1)])
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=30, batch_size=4, verbose=0)

print("LSTM Training Completed.")

test_input = scaled[-3:].reshape(1,3,1)
prediction = model.predict(test_input, verbose=0)
predicted_price = prediction[0][0]*(maximum-minimum)+minimum

print("Predicted Next Closing Price:", round(float(predicted_price), 2))
