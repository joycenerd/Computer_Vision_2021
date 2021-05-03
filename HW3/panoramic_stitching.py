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


class Match_node:
    def __init__(self, first_idx, dist, sec_idx):
        self.first_idx = int(first_idx)
        self.dist = dist
        self.sec_idx = int(sec_idx)


def get_ratio_matches(desc1, desc2):
    # two lowest distance as potential match
    dist = scipy.spatial.distance.cdist(desc1, desc2, metric="euclidean")
    matches = np.empty((desc1.shape[0], 2), dtype=object)
    for i in range(desc1.shape[0]):
        idx = [j for j in range(desc2.shape[0])]
        dist_array = np.array(list(zip(dist[i, :], idx)))
        sorted_dist = dist_array[np.argsort(dist_array[:, 0])]
        match1 = Match_node(i, sorted_dist[0][1], sorted_dist[0][0])
        match2 = Match_node(i, sorted_dist[1][1], sorted_dist[1][0])
        matches[i][0] = match1
        matches[i][1] = match2
    return matches


def get_good_matches(matches):
    # ratio test to confirm the matches
    good_matches = []
    for [a, b] in matches:
        if a.dist < 0.5 * b.dist:
            good_matches.append(a)
    good_matches = np.array(good_matches)
    return good_matches


class DMatch:
    def __init__(self, match):
        self.distance = match.dist
        self.trainIdx = match.first_idx
        self.queryIdx = match.sec_idx
        self.imgIdx = 0


def feature_matching(desc1, desc2, p1, p2):
    matches = get_ratio_matches(desc1, desc2)
    print(f"Initail matches number: {matches.shape[0]}")
    good_matches = get_good_matches(matches)
    print(f"Good matches number: {good_matches.shape[0]}")
    matcher_obj = np.empty(good_matches.shape[0], dtype=object)
    kp1 = np.empty(good_matches.shape[0], dtype=object)
    kp2 = np.empty(good_matches.shape[0], dtype=object)
    for i, match in enumerate(good_matches):
        matcher_obj[i] = DMatch(match)
        kp1[i] = p1[match.first_idx]
        kp2[i] = p2[match.sec_idx]
    return matcher_obj, kp1, kp2


if __name__ == "__main__":
    image1 = cv2.imread(DATA_PATH + "/hill1.jpg")
    image2 = cv2.imread(DATA_PATH + "/hill2.jpg")
    points1, descriptors1, points2, descriptors2 = detect_interest_p(image1, image2)
    good_matches, kp1, kp2 = feature_matching(
        descriptors1, descriptors2, points1, points2
    )
    match_img = cv2.drawMatchesKnn(
        image1, kp1, image2, kp2, good_matches, None, flags=2
    )
    match_img = cv2.cvtColor(cv2.BGR2RGB)
    plt.imshow(match_img)
    plt.show()
