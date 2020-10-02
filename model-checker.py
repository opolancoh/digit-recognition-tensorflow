import matplotlib.image as mpimg

from skimage.color import rgb2gray
from skimage.transform import resize
from tensorflow.keras.models import load_model
from utils import preprocess_image

img_rows, img_cols = 28, 28

model_list = ["model-1.h5", "model-2.h5"]

for model_name in model_list:
    print(f"Checking model '{model_name}'")
    model = load_model(model_name)
    predictions = 0
    for i in range(10):
        img = preprocess_image(f"data/validation/{i}-28x28.png", img_rows, img_cols)
        # reshape the image
        img = img.reshape(1, img_rows, img_cols, 1)
        # normalize image
        img = img / 255
        # predict digit
        prediction = model.predict(img)
        predicted = prediction.argmax()
        success = i == predicted
        prediction_text = "no"
        if success:
            predictions += 1
            prediction_text = "yes"
        print(f"{i} -> {predicted} {prediction_text}")
    print(f"Prediction result: {(predictions/10)*100} %")
    print("")
