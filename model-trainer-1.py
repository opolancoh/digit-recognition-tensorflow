# https://www.sitepoint.com/keras-digit-recognition-tutorial/

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Digit dataset
import dataset

# x (image), y (label)
(x_train, y_train), (x_test, y_test) = dataset.load_data()

# To work with the Keras API, we need to reshape each image to the format of (M x N x 1)
img_rows, img_cols = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# Normalize the image data by dividing each pixel value by 255 (since RGB value can range from 0 to 255)
x_train = x_train / 255
x_test = x_test / 255

# Convert the dependent variable in the form of integers to a binary class matrix
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# From here we are ready to create the model and train it
# Initialize a sequential model
model = Sequential(
    [
        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(img_rows, img_cols, 1),
        ),
        Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        # Max-pooling layer, using 2x2 pool size
        MaxPooling2D(pool_size=(2, 2)),
        # Drop 25% of the units
        Dropout(rate=0.25),
        # Add a flattening layer to convert the previous hidden layer into a 1D array
        Flatten(),
        # Add hidden layers
        Dense(units=128, activation="relu"),
        Dropout(0.5),
        # Softmax activation is used when we’d like to classify the data into a number of pre-decided classes
        Dense(units=num_classes, activation="softmax"),
    ]
)

# Compile and train model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(
    x=x_train,
    y=y_train,
    batch_size=128,
    epochs=10,
    verbose=1,
    validation_data=(x_test, y_test),
)

# Evaluate
model.evaluate(x=x_test, y=y_test, verbose=2)

# Save model to file
filename = "model-1.h5"
model.save(filename)
print(f"Model saved to {filename}")
