import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np


DATA_PATH = "./hw2_data/task3_colorizing"


def ssd(image1, image2):
    # sqrt((a-b)^2)
    height, width, channel = image2.shape
    image1 = np.array(image1).flatten()
    image2 = np.array(image2).flatten()
    dist = np.sqrt(np.sum((image1-image2)**2))/height*width
    return dist


if __name__ == "__main__":
    for image_path in glob.glob(DATA_PATH+"/*"):
        # read the image
        image = cv2.imread(image_path)
        print(image_path)

        # split the image to B,G,R three channel
        height, width, channel = image.shape
        print(f'height: {height}, width: {width}')
        height = int(height/3)
        B = image[0:height, 0:width]
        G = image[height:2*height, 0:width]
        R = image[2*height:3*height, 0:width]

        # cropped the border of the image
        crop_height = int(height*0.05)
        crop_width = int(width*0.05)
        B = B[crop_height:height-crop_height, crop_width:width-crop_width]
        G = G[crop_height:height-crop_height, crop_width:width-crop_width]
        R = R[crop_height:height-crop_height, crop_width:width-crop_width]
        plt.imshow(B)
        plt.show()
        break

        # SSD
