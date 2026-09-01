import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/Users/rohith/Desktop/COMPUTER VISION/Screenshot 2026-07-20 at 11.36.06 AM.png')

if img is None:
    print("Error: Could not load image. Please check the file path.")
else:
    b, g, r = cv2.split(img)
    rgb_img = cv2.merge([r, g, b])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((2, 2), np.uint8)

    closing = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    sure_bg = cv2.dilate(
        closing, kernel, iterations=3
    )

    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.imshow(rgb_img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(thresh, cmap='gray')
    plt.title("Otsu's Threshold")
    plt.axis('off')

    plt.subplot(234)
    plt.imshow(closing, cmap='gray')
    plt.title("MorphologyEx: Closing")
    plt.axis('off')

    plt.subplot(235)
    plt.imshow(sure_bg, cmap='gray')
    plt.title("Dilation")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    plt.imsave('dilation.png', sure_bg)
    print("Dilation image saved successfully.")
