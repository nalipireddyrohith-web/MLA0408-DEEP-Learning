import numpy as np
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.models import Model

np.random.seed(42)
image = np.random.rand(1,224,224,3).astype("float32")
num_classes = 5

base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False

x = Conv2D(128, (3,3), padding="same", activation="relu")(base_model.output)
x = UpSampling2D(size=(8,8))(x)
output = Conv2D(num_classes, (1,1), padding="same", activation="softmax")(x)

model = Model(base_model.input, output)
prediction = model.predict(image, verbose=0)
segmentation = np.argmax(prediction[0], axis=-1)

print("Input Image Shape:", image.shape)
print("Segmentation Output Shape:", segmentation.shape)
print("\nPixel-wise segmentation completed.")
print("Sample Pixel Classes:")
print(segmentation[100:105,100:105])
