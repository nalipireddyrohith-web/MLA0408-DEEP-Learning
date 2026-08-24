import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

np.random.seed(42)
tf.random.set_seed(42)

X = np.random.rand(20,224,224,3).astype("float32")
y = np.array([0,1,0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,1,0])

base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(2, activation="softmax")(x)

model = Model(base_model.input, output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=2, batch_size=4, verbose=1)

test_image = np.random.rand(1,224,224,3).astype("float32")
prediction = model.predict(test_image, verbose=0)
classes = ["Benign", "Malignant"]
result = np.argmax(prediction)

print("\nSkin Cancer Prediction")
print("Result:", classes[result])
print("Confidence:", round(float(prediction[0][result])*100, 2), "%")
