import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import math
import argparse


DATA_PATH = "./my_data/"
SAVE_PATH = "./results/"


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


def get_good_matches(matches, threshold):
    # ratio test to confirm the the match
    good_matches = []
    for [a, b] in matches:
        if a.distance < threshold * b.distance:
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
        x, y = p1[i]
        x_prime, y_prime = p2[i]
        H[2*1, :] = [x, y, 1, 0, 0, 0, -x*x_prime, -y*x_prime, -x_prime]
        H[2*i+1, :] = [0, 0, 0, x, y, 1, -x*y_prime, -y*y_prime, -y_prime]

    # solve PH=0, H is the last column of v
    # u, s, vh = np.linalg.svd(H) # vh (9,9)
    #homography = (vh.T[:,-1] / vh[-1,-1]).reshape(3, 3)
    homography, _ = cv2.findHomography(p1, p2)
    # homography=np.array(homography)
    # print(homography)
    # print(homography.shape)
    return homography


def count_inliers(correspondence, homography):
    inliers = 0
    for i in range(correspondence.shape[0]):
        src = [correspondence[i][0], correspondence[i][1], 1]
        dest = [correspondence[i][2], correspondence[i][3], 1]
        src = np.array(src)
        dest = np.array(dest)
        pred = homography@src.T
        pred /= pred[2]
        loss = np.sum(abs(pred-dest))
        if loss < 1:
            inliers += 1
    return inliers


def RANSAC(correspondence):
    max_inliers = 0
    j = 0
    for i in range(1000):
        rand_choice = list(np.random.random(size=4) * correspondence.shape[0])
        rand_choice = list(map(int, rand_choice))
        coor1 = correspondence[rand_choice[0]]
        coor2 = correspondence[rand_choice[1]]
        coor3 = correspondence[rand_choice[2]]
        coor4 = correspondence[rand_choice[3]]
        homography = find_homography(coor1, coor2, coor3, coor4)
        inliers = count_inliers(correspondence, homography)
        if inliers > max_inliers:
            best_H = homography
            max_inliers = inliers
    print(max_inliers)
    print("Best homography:")
    print(best_H)
    return homography


def feature_matching(desc1, desc2, p1, p2, img1, img2, threshold, img_name):
    matches = get_ratio_matches(desc1, desc2)
    print(f"Number of initial matches: {len(matches)}")
    good_matches = get_good_matches(matches, threshold)
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
    plt.savefig(SAVE_PATH + img_name+"_feature_matching.jpg")
    plt.show()
    return correspondence


def decide_out_size(img1, img2, homography):
    four_corner = np.zeros((4, 3))
    four_corner[0, :] = [0, 0, 1]
    four_corner[1, :] = [img1.shape[1], 0, 1]
    four_corner[2, :] = [0, img1.shape[0], 1]
    four_corner[3, :] = [img1.shape[1], img1.shape[0], 1]
    min_x = 0
    min_y = 0
    max_y, max_x, _ = img2.shape
    for corner in four_corner:
        trans_corner = homography@corner.T
        trans_corner /= trans_corner[2]
        x, y, _ = trans_corner
        min_x = min(min_x, math.floor(x))
        min_y = min(min_y, math.floor(y))
        max_x = max(max_x, math.ceil(x))
        max_y = max(max_y, math.ceil(y))
    return min_x, min_y, max_x, max_y


def bilinear_interpolation(x, y, img1, img2):
    h, w, _ = img1.shape
    x1 = math.floor(x)
    if x1 < 0:
        x1 = 0
    y1 = math.floor(y)
    if y1 < 0:
        y1 = 0
    x2 = math.ceil(x)
    if x2 >= w:
        x2 = w-1
    y2 = math.ceil(y)
    if y2 >= h:
        y2 = h-1

    q11 = img1[y1, x1, :]
    q21 = img1[y1, x2, :]
    q12 = img1[y2, x1, :]
    q22 = img1[y1, x1, :]

    if x1 == x2 and y1 == y2:
        rgb = q11
    elif x1 == x2 and y1 != y2:
        rgb = (q11 * (y2 - y) + q12 * (y - y1))/(y2-y1+0.0)
    elif y1 == y1 and x1 != x2:
        rgb = (q21 * (x - x1) + q22 * (x2-x))/(x2-x1+0.0)
    else:
        rgb = (q11 * (x2 - x) * (y2 - y) +
               q21 * (x - x1) * (y2 - y) +
               q12 * (x2 - x) * (y - y1) +
               q22 * (x - x1) * (y - y1)
               ) / ((x2 - x1) * (y2 - y1)+0.0)
    return rgb


def img_morphing(offset_x, img1, x):
    left_border = offset_x
    right_border = img1.shape[1]
    ratio1 = (right_border-x)/(right_border-left_border+0.0)
    ratio2 = (x-left_border)/(right_border-left_border+0.0)
    return ratio1, ratio2


def img_warping(img1, img2, homography, img_name):
    min_x, min_y, max_x, max_y = decide_out_size(img1, img2, homography)
    if min_x < 0:
        offset_x = -min_x
    else:
        offset_x = 0
    if min_y < 0:
        offset_y = -min_y
    else:
        offset_y = 0
    w = max_x+offset_x
    h = max_y+offset_y
    out_image = np.full((h, w, 3), 0)
    existed = np.zeros((h, w))
    # move image 2 to output image
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            y = offset_y+i
            x = offset_x+j
            out_image[y, x, :] = img2[i, j, :]
            existed[y, x] = 1
    tmp_image = np.float32(out_image/255.0)
    tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)
    #plt.figure(figsize=(20, 10))
    # plt.imshow(tmp_image)
    # plt.show()

    # move image 1 to output image
    h_inv = np.linalg.inv(homography)
    for i in range(h):
        for j in range(w):
            p2 = np.array([j-offset_x, i-offset_y, 1])
            p1 = h_inv@p2.T
            p1 /= p1[-1]
            x, y, _ = p1
            if x < 0 or x >= img1.shape[1] or y < 0 or y >= img1.shape[0]:
                continue
            elif j < offset_x or i < offset_y or i >= offset_y+img2.shape[0]:
                out_image[i, j, :] = bilinear_interpolation(x, y, img1, img1)
            else:
                img1_rgb = bilinear_interpolation(x, y, img1, img1)
                ratio1, ratio2 = img_morphing(offset_x, img1, x)
                # print(ratio1+ratio2)
                out_image[i, j, :] = img1_rgb*ratio1+out_image[i, j, :]*ratio2

            # else:
    out_image = np.float32(out_image/255.0)
    out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 10))
    plt.imshow(out_image)
    plt.savefig(SAVE_PATH+img_name+"_panorama.jpg")
    plt.show()
    return out_image


def crop_image(image, ratio):
    height, width, channel = image.shape
    crop_height = int(height * ratio)
    crop_width = int(width * ratio)
    cropped_image = image[crop_height: height -
                          crop_height, crop_width: width - crop_width]
    return cropped_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_name", type=str, default="hill",
                        help="Your image name before the number of the image")
    parser.add_argument("--dist_thres", type=float, default=0.2,
                        help="threshold of ratio test in feature matching")
    args = parser.parse_args()
    image1 = cv2.imread(DATA_PATH + args.image_name+"1.jpg")
    image2 = cv2.imread(DATA_PATH+args.image_name+"2.jpg")
    image1 = crop_image(image1, 0.03)
    image2 = crop_image(image2, 0.03)
    points1, descriptors1, points2, descriptors2 = detect_interest_p(
        image1, image2)
    correspondence = feature_matching(
        descriptors1, descriptors2, points1, points2, image1, image2, args.dist_thres, args.image_name
    )
    homography = RANSAC(correspondence)
    out_image = img_warping(image1, image2, homography, args.image_name)

    # python panoramic_stitching.py --image_name hill --dist_thres 0.2
    # python panoramic_stitching.py --image_name S --dist_thres 0.3
    # python panoramic_stitching.py --image_name tv --dist_thres 0.5
    # python panoramic_stitching.py --image_name stele --dist_thres 0.4
    # python .\panoramic_stitching.py --image_name pond --dist_thres 0.6
