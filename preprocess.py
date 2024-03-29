import numpy as np
import pandas as pd
from scipy.ndimage import affine_transform
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def add_transform(data):
    # Define the transformation parameters
    scale = np.random.uniform(.9, 1.7)  # Random scaling factor between 0.8 and 1.2
    translate_x = np.random.uniform(-5, 5)  # Random x-axis translation between -2 and 2 pixels
    translate_y = np.random.uniform(-5, 5)  # Random y-axis translation between -2 and 2 pixels
    # Define the affine transformation matrix
    matrix = [[scale, 0, translate_x],
              [0, scale, translate_y],
              [0, 0, 1]]
    # Apply the affine transformation
    transformed_image = affine_transform(data, matrix, cval=0)
    return transformed_image


def add_noise(image, amount=0.015, salt_vs_pepper_ratio=0.5):
    noisy_image = np.copy(image)

    # Add salt noise
    num_salt = np.ceil(amount * image.size * salt_vs_pepper_ratio)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords] = 255

    # Add pepper noise
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper_ratio))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords] = 0

    return noisy_image


# Preprocess the data
def preprocess_data(path, noise=0, transform=0):
    data = pd.read_csv(path)
    my_data = data.values
    x = my_data[:, 1:]
    y = my_data[:, :1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Reshape to 28*28 pixels
    x_train = np.reshape(x_train, (x_train.shape[0], 28, 28))
    x_test = np.reshape(x_test, (x_test.shape[0], 28, 28))

    # Add noise and transformation if needed
    if noise:
        for i in range(x_train.shape[0]):
            x_train[i] = add_noise(x_train[i])
        for i in range(x_test.shape[0]):
            x_test[i] = add_noise(x_test[i])

    if transform:
        for i in range(x_train.shape[0]):
            x_train[i] = add_transform(x_train[i])
        for i in range(x_test.shape[0]):
            x_test[i] = add_transform(x_test[i])

    # Reshape the train & test image dataset
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    # Downsample the values
    x_train = x_train / 255.
    x_test = x_test / 255.

    # Convert  single int values to categorical values
    y_train = to_categorical(y_train, num_classes=26)
    y_test = to_categorical(y_test, num_classes=26)

    return x_train, y_train, x_test, y_test
