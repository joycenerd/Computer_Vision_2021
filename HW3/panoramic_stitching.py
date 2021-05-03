import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial


DATA_PATH = "./data"
SAVE_PATH = "./results"


def detect_interest_p(image1, image2):
    feature_type = cv2.SIFT_create()
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    points1, descriptors1 = feature_type.detectAndCompute(gray1, None)
    points2, descriptors2 = feature_type.detectAndCompute(gray2, None)
    return points1, descriptors1, points2, descriptors2


class DMatch:
    def __init__(self, distance, trainIdx, queryIdx):
        self.distance = distance
        self.trainIdx = int(trainIdx)
        self.queryIdx = int(queryIdx)
        self.imgIdx = 0


def get_ratio_matches(desc1, desc2):
    # two lowest distance as potential match
    dist = scipy.spatial.distance.cdist(desc1, desc2, metric="euclidean")
    matches = []
    for i in range(desc1.shape[0]):
        match = []
        idx = [j for j in range(desc2.shape[0])]
        dist_array = np.array(list(zip(dist[i, :], idx)))
        sorted_dist = dist_array[np.argsort(dist_array[:, 0])]
        for j in range(2):
            match.append(DMatch(sorted_dist[j][0], i, sorted_dist[j][1]))
        matches.append(match)
    return matches


def get_good_matches(matches):
    # ratio test to confirm the the match
    good_matches = []
    for [a, b] in matches:
        if a.distance < 0.3 * b.distance:
            good_matches.append(a)
    return good_matches


def draw_matching(good_matches, kp1, kp2, img1, img2):
    # draw circle and line
    match_img = np.concatenate((img1, img2), axis=1)
    h1, w1, c1 = img1.shape
    for i in range(kp1.shape[0]):
        color = list(np.random.random(size=3) * 256)
        coor1 = list(kp1[i])
        x1 = int(coor1[0])
        y1 = int(coor1[1])
        match_img = cv2.circle(match_img, (x1, y1), 5, color, 1)
        coor2 = list(kp2[i])
        x2 = int(w1 + coor2[0])
        y2 = int(coor2[1])
        match_img = cv2.circle(match_img, (x2, y2), 5, color, 1)
        match_img = cv2.line(match_img, (x1, y1), (x2, y2), color, 1)

    match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
    return match_img


def find_homography(coor1, coor2, coor3, coor4):
    H = np.zeros((8, 9))
    p1 = np.zeros((4, 2))
    p2 = np.zeros((4, 2))
    p1[0, :] = [coor1[0], coor1[1]]
    p2[0, :] = [coor1[2], coor1[3]]
    p1[1, :] = [coor2[0], coor2[1]]
    p2[1, :] = [coor2[2], coor2[3]]
    p1[2, :] = [coor3[0], coor3[1]]
    p2[2, :] = [coor3[2], coor3[3]]
    p1[3, :] = [coor4[0], coor4[1]]
    p2[3, :] = [coor4[2], coor4[3]]
    for i in range(4):
        H[i, :] = [
            p1[i][1],
            p1[i][0],
            1,
            0,
            0,
            0,
            -p2[i][1] * p1[i][1],
            -p2[i][1] * p1[i][0],
            -p2[i][1],
        ]
        H[2 * i + 1, :] = [
            0,
            0,
            0,
            p1[i][1],
            p1[i][0],
            1,
            -p2[i][0] * p1[i][1],
            -p2[i][0] * p1[i][0],
            -p2[i][0],
        ]
        [U, S, V] = np.linalg.svd(H)
        homography = V[-1, :]
        homography = np.reshape(homography, (3, 3))
        return homography


def RANSAC(correspondence):
    for i in range(1000):
        rand_choice = list(np.random.random(size=4) * correspondence.shape[0])
        rand_choice = list(map(int, rand_choice))
        coor1 = correspondence[rand_choice[0]]
        coor2 = correspondence[rand_choice[1]]
        coor3 = correspondence[rand_choice[2]]
        coor4 = correspondence[rand_choice[3]]
        homography = find_homography(coor1, coor2, coor3, coor4)
        print(homography)
        break


def feature_matching(desc1, desc2, p1, p2, img1, img2):
    matches = get_ratio_matches(desc1, desc2)
    print(f"Number of initial matches: {len(matches)}")
    good_matches = get_good_matches(matches)
    print(f"Number of good matches: {len(good_matches)}")
    kp1 = np.empty(len(good_matches), dtype=tuple)
    kp2 = np.empty(len(good_matches), dtype=tuple)
    correspondence = np.zeros((len(good_matches), 4))
    for i, match in enumerate(good_matches):
        kp1[i] = p1[match.trainIdx].pt
        kp2[i] = p2[match.queryIdx].pt
        correspondence[i, :] = [kp1[i][0], kp1[i][1], kp2[i][0], kp2[i][1]]

    match_img = draw_matching(good_matches, kp1, kp2, img1, img2)
    plt.figure(figsize=(20, 10))
    plt.imshow(match_img)
    plt.savefig(SAVE_PATH + "/hill_feature_matching.jpg")
    plt.show()
    return correspondence


if __name__ == "__main__":
    image1 = cv2.imread(DATA_PATH + "/hill1.jpg")
    image2 = cv2.imread(DATA_PATH + "/hill2.jpg")
    points1, descriptors1, points2, descriptors2 = detect_interest_p(image1, image2)
    correspondence = feature_matching(
        descriptors1, descriptors2, points1, points2, image1, image2
    )
    RANSAC(correspondence)
