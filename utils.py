import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage.color import rgb2gray


def preprocess_image(fname, img_rows, img_cols):
    try:
        img = mpimg.imread(fname)
        img_resized = resize(
            img, (img_rows, img_cols), mode="constant", anti_aliasing=True
        )
        img_gray = rgb2gray(img_resized,)
        # plt.imshow(img_gray, cmap=plt.cm.gray)
        # plt.show()
        return img_gray
    except IOError:
        print(f"'{fname}' is not an image file")
        return None
