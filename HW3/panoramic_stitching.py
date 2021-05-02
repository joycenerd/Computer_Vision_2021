import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial


DATA_PATH = "./data"


def detect_interest_p(image1, image2):
    feature_type = cv2.SIFT_create()
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    points1, descriptors1 = feature_type.detectAndCompute(gray1, None)
    points2, descriptors2 = feature_type.detectAndCompute(gray2, None)
    return points1, descriptors1, points2, descriptors2


def feature_matching(desc1, desc2):
    dist = scipy.spatial.distance.cdist(desc1, desc2, metric="euclidean")
    print(dist)
    print(dist.shape)


if __name__ == "__main__":
    image1 = cv2.imread(DATA_PATH + "/hill1.jpg")
    image2 = cv2.imread(DATA_PATH + "/hill2.jpg")
    points1, descriptors1, points2, descriptors2 = detect_interest_p(image1, image2)
    feature_matching(descriptors1, descriptors2)
