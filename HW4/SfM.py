from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--img", type=str, default="Mesona", help="Which set of image do you use")
parser.add_argument(
    "--ratio", type=float, default=0.3, help="the ratio for ratio test used for finding good feature matching"
)
args = parser.parse_args()


DATA_PATH = "./data/"


def read_intrinsic():
    if args.img == "Mesona":
        K1 = np.array([1.4219, 0.0005, 0.509, 0, 1.4219, 0.3802, 0, 0, 0.001])
        K2 = K1
    elif args.img == "Statue":
        K1 = np.array(
            [5426.566895, 0.678017, 330.096680, 0.000000, 5423.133301, 648.950012, 0.000000, 0.000000, 1.000000]
        )
        K2 = np.array(
            [5426.566895, 0.678017, 387.430023, 0.000000, 5423.133301, 620.616699, 0.000000, 0.000000, 1.000000]
        )

    # The last element should be 1 and reshape to (3,3)
    K1 /= K1[-1]
    K1 = K1.reshape((3, 3))

    K2 /= K2[-1]
    K2 = K2.reshape((3, 3))

    return K1, K2


def detect_interest_p(img):
    feat_type = cv2.SIFT_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p, desc = feat_type.detectAndCompute(gray, None)
    return p, desc


class DMatch:
    def __init__(self, distance, trainIdx, queryIdx):
        self.distance = distance
        self.trainIdx = int(trainIdx)
        self.queryIdx = int(queryIdx)
        self.imgIdx = 0


def get_ratio_matches(desc1, desc2):
    dist = cdist(desc1, desc2, metric="euclidean")
    matches = np.empty((desc1.shape[0], 2), dtype=object)
    for i in range(desc1.shape[0]):
        idx = [j for j in range(desc2.shape[0])]
        dist_array = np.array(list(zip(dist[i, :], idx)))
        sorted_dist = dist_array[np.argsort(dist_array[:, 0])]
        for j in range(2):
            matches[i, j] = DMatch(sorted_dist[j, 0], i, sorted_dist[j, 1])

    return matches


def get_good_matches(matches):
    # ration test
    good_matches = []
    for a, b in matches:
        if a.distance < args.ratio * b.distance:
            good_matches.append(a)
    return good_matches


def draw_matching(kp1, kp2, img1, img2):
    match_img = np.concatenate((img1, img2), axis=1)  # output matching image
    h1, w1, c1 = img1.shape
    for i in range(kp1.shape[0]):
        # locate matching point
        color = list(np.random.random(size=3) * 256)
        coor1 = list(kp1[i])
        x1 = int(coor1[0])
        y1 = int(coor1[1])
        coor2 = list(kp2[i])
        x2 = w1 + int(coor2[0])
        y2 = int(coor2[1])

        # draw circle on matching points and line between them
        match_img = cv2.circle(match_img, (x1, y1), 5, color, 1)
        match_img = cv2.circle(match_img, (x2, y2), 5, color, 1)
        match_img = cv2.line(match_img, (x1, y1), (x2, y2), color, 1)

    return match_img


def find_correspondence(img1, img2):
    # get interest points
    p1, desc1 = detect_interest_p(img1)
    p2, desc2 = detect_interest_p(img2)
    # feature matching
    matches = get_ratio_matches(desc1, desc2)
    good_matches = get_good_matches(matches)
    # store the matching keypoints
    kp1 = np.empty(len(good_matches), dtype=tuple)
    kp2 = np.empty(len(good_matches), dtype=tuple)
    correspondence = np.zeros((len(good_matches), 4), dtype=float)
    for i, match in enumerate(good_matches):
        kp1[i] = p1[match.trainIdx].pt
        kp2[i] = p2[match.queryIdx].pt
        correspondence[i, :] = [kp1[i][0], kp1[i][1], kp2[i][0], kp2[i][1]]

    # draw feature matching
    match_img = draw_matching(kp1, kp2, img1, img2)
    match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 5))
    plt.imshow(match_img)
    plt.axis("off")
    plt.show()

    return correspondence


if __name__ == "__main__":
    img1 = cv2.imread(DATA_PATH + "Mesona1.JPG")
    img2 = cv2.imread(DATA_PATH + "Mesona2.JPG")
    K1, K2 = read_intrinsic()
    print("Intrinsic matrix of K1:")
    print(K1)
    print("")
    print("Intrinsic matrix of K2:")
    print(K2)
    print("")
    correspondence = find_correspondence(img1, img2)
    print(f"Number of correspondence: {correspondence.shape[0]}")
