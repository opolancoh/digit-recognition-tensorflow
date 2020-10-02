# http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

import os
import numpy as np

from tqdm import tqdm
from utils import preprocess_image


img_rows, img_cols = 28, 28


def get_data_from_images(path):
    valid_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    x = []
    y = []

    for dir_name in tqdm(os.listdir(path)):
        if dir_name in valid_classes:
            file_count = 0
            class_path = os.path.join(path, dir_name)
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                img = preprocess_image(file_path, img_rows, img_cols)
                img_arr = np.array(img)
                x.append(img_arr)
                y.append(dir_name)
                file_count += 1

    x = np.array(x)
    y = np.array(y)
    return x, y


def load_data():
    train_path = "data/train/"
    test_path = "data/test/"

    print(f"\nGetting data from '{train_path}' ...")
    x_train, y_train = get_data_from_images(train_path)
    print(f"\nGetting data from '{test_path}' ...")
    x_test, y_test = get_data_from_images(test_path)

    return (x_train, y_train), (x_test, y_test)
