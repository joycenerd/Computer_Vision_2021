import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import re


DATA_PATH = "./hw2_data/task3_colorizing"


def ssd(image1, image2):
    # sqrt((a-b)^2)
    height, width, channel = image2.shape
    image1 = np.array(image1).flatten()
    image2 = np.array(image2).flatten()
    dist = np.sqrt(np.sum((image1 - image2) ** 2)) / height * width
    return dist


def crop_image(image, ratio):
    height, width, channel = image.shape
    crop_height = int(height * ratio)
    crop_width = int(width * ratio)
    cropped_image = image[crop_height : height - crop_height, crop_width : width - crop_width]
    return cropped_image


def shift_im(image, offset):
    # shift the image and replace the border with original border
    height, width, channel = image.shape
    shifted_image = np.roll(image, offset, axis=(0, 1))
    if offset[0] < 0:  # shift up
        border = image[-1, :]
        for i in range(abs(offset[0])):
            shifted_image[height - i - 1, :] = border
    elif offset[0] >= 0:  # shift down
        border = image[0, :]
        for i in range(abs(offset[0])):
            shifted_image[i, :] = border
    if offset[1] < 0:  # shift left
        border = image[:, -1]
        for i in range(abs(offset[1])):
            shifted_image[:, width - i - 1] = border
    elif offset[1] >= 0:  # shift right
        border = image[:, 0]
        for i in range(abs(offset[1])):
            shifted_image[:, i] = border
    return shifted_image


def naive_search(image1, image2):
    # Use ssd to decide the range to shift image2
    # let R,B align with G
    min_score = float("inf")
    horiz_offset = 0
    vert_offset = 0
    # shift vertical
    for i in range(-20, 21):
        for j in range(-20, 21):
            shifted_image_2 = shift_im(image2, [i, j])
            score = ssd(image1, shifted_image_2)
            if score < min_score:
                vert_offset = i
                horiz_offset = j
                min_score = score
    aligned_image = shift_im(image2, [vert_offset, horiz_offset])
    return aligned_image, vert_offset, horiz_offset


if __name__ == "__main__":
    for image_path in glob.glob(DATA_PATH + "/*.jpg"):
        # read the image
        image_name = image_path.split("\\")[1]
        image = cv2.imread(image_path)
        print(image_path)

        # split the image to B,G,R three channel
        height, width, channel = image.shape
        print(f"height: {height}, width: {width}")
        height = int(height / 3)
        B = image[0:height, 0:width]
        G = image[height : 2 * height, 0:width]
        R = image[2 * height : 3 * height, 0:width]

        # cropped the border of the image
        cropped_B = crop_image(B, 0.08)
        cropped_G = crop_image(G, 0.08)
        cropped_R = crop_image(R, 0.08)
        height, width, channel = cropped_G.shape
        unaligned_image = cv2.merge((cropped_B[:, :, 0], cropped_G[:, :, 0], cropped_R[:, :, 0]))

        # SSD
        align_B, B_vert_offset, B_horiz_offset = naive_search(cropped_G, cropped_B)
        align_R, R_vert_offet, R_horiz_offset = naive_search(cropped_G, cropped_R)
        aligned_image = cv2.merge((align_B[:, :, 0], cropped_G[:, :, 0], align_R[:, :, 0]))

        # plot
        plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(unaligned_image)
        plt.title("Unaligned")
        plt.subplot(1, 2, 2)
        plt.title("Aligned")
        plt.imshow(aligned_image)
        caption = (
            f"B align offset: [{B_horiz_offset}, {B_vert_offset}], R align offset: [{R_horiz_offset}, {R_vert_offet}]"
        )
        plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=12)
        plt.savefig("./results/task3/" + image_name)
        plt.show()