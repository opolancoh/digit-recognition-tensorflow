# https://cs50.harvard.edu/ai/2020/

import tensorflow as tf

# Digit dataset
import dataset

# Prepare data for training
(x_train, y_train), (x_test, y_test) = dataset.load_data()

# Normalize the image data by dividing each pixel value by 255 (since RGB value can range from 0 to 255)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert the dependent variable in the form of integers to a binary class matrix
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Reshape to be [samples][width][height][channels]
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# Create a convolutional neural network
model = tf.keras.models.Sequential(
    [
        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Flatten units
        tf.keras.layers.Flatten(),
        # Add a hidden layer with dropout
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        # Add an output layer with output units for all 10 digits
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

# Train neural network
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10)

# Evaluate neural network performance
model.evaluate(x_test, y_test, verbose=2)

# Save model to file
filename = "model-2.h5"
model.save(filename)
print(f"Model saved to {filename}")
