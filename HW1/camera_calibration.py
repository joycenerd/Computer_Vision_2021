import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image


def is_positive_definite(A):
    try:
        _ = np.linalg.cholesky(A)
        return True
    except np.LinAlgError:
        return False


if __name__ == '__main__':
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # (8,6) is for the given testing images.
    # If you use the another data (e.g. pictures you take by your smartphone),
    # you need to set the corresponding numbers.
    corner_x = 7
    corner_y = 7
    objp = np.zeros((corner_x * corner_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # points on object plane
    imgpoints = []  # points in image plane.

    # Make a list of calibration images
    images = glob.glob('data/*.jpg')

    # Step through the list and search for chessboard corners
    print('Start finding chessboard corners...')
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray)

        # Find the chessboard corners
        print('find the chessboard corners of', fname)
        ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)  # points on object plane
            imgpoints.append(corners)  # points on image plane

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (corner_x, corner_y), corners, ret)
            plt.imshow(img)
            # plt.show()

    #######################################################################################################
    #                                Homework 1 Camera Calibration                                        #
    #               You need to implement camera calibration(02-camera p.76-80) here.                     #
    #   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
    #                                          H I N T                                                    #
    #                        1.Use the points in each images to find Hi                                   #
    #                        2.Use Hi to find out the intrinsic matrix K                                  #
    #                        3.Find out the extrinsics matrix of each images.                             #
    #######################################################################################################
    """
    print('Camera calibration...')
    img_size = (img.shape[1], img.shape[0])
    # You need to comment these functions and write your calibration function from scratch.
    # Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
    # In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    Vr = np.array(rvecs)
    Tr = np.array(tvecs)
    extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
    """
    # Write your code here

    # get homography matrix #################################
    objpoints = np.array(objpoints)
    imgpoints = np.array(imgpoints)
    coeff_mat = []
    for i in range(objpoints.shape[0]):
        each_objpoints = objpoints[i]
        each_imgpoints = imgpoints[i]
        objpoints[:, 2] = 1
        extra_col = np.ones((objpoints.shape[1], 1))
        each_imgpoints = np.squeeze(each_imgpoints, 1)
        each_imgpoints = np.append(each_imgpoints, extra_col, axis=1)
        h, status = cv2.findHomography(each_objpoints, each_imgpoints)

        # get B #################################

        # get the coefficient of the B matrix by h1.T*B*h2=0
        coeff = [
            h[1, 0] * h[0, 0],
            h[1, 0] * h[0, 1] + h[0, 0] * h[1, 1],
            h[1, 0] * h[0, 2] + h[1, 2] * h[0, 0],
            h[1, 1] * h[0, 1],
            h[1, 1] * h[0, 2] + h[1, 2] * h[0, 1],
            h[1, 2] * h[0, 2]
        ]
        coeff_mat.append(coeff)
    coeff_mat = np.array(coeff_mat)

    # solve B by svd
    U, sig, Vt = np.linalg.svd(coeff_mat)
    V = Vt.T
    B = V[:, -1]

    B_mat = np.array([B[0], B[1], B[2],
                      B[1], B[3], B[4],
                      B[2], B[4], B[5]])
    B_mat = B_mat.reshape(3, 3)
    # print(B_mat)

    # convert B to positive semi-definite
    BB = (B_mat + B_mat.T) / 2
    _, s, V = np.linalg.svd(BB)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    B2 = (BB + H) / 2
    B3 = (B2 + B2.T) / 2
    if not is_positive_definite(B3):
        print("yes")
        spacing = np.spacing(np.linalg.norm(B))
        I = np.eye(B_mat.shape[0])
        k = 1
        while not is_positive_definite(B3):
            min_eig = np.min(np.real(np.linalg.eigvals(B3)))
            B3 += I * (-min_eig * k ** 2 + spacing)
            k += 1
    B_mat = B3
    print(B_mat)

    B_inv = np.linalg.inv(B_mat)
    print(B_inv)
    K = np.linalg.cholesky(B_inv)
    print(K)

    """
    # show the camera extrinsics
    print('Show the camera extrinsics')
    # plot setting
    # You can modify it for better visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    # camera setting
    camera_matrix = mtx
    cam_width = 0.064/0.1
    cam_height = 0.032/0.1
    scale_focal = 1600
    # chess board setting
    board_width = 8
    board_height = 6
    square_size = 1
    # display
    # True -> fix board, moving cameras
    # False -> fix camera, moving boards
    min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                    scale_focal, extrinsics, board_width,
                                                    board_height, square_size, True)
    
    X_min = min_values[0]
    X_max = max_values[0]
    Y_min = min_values[1]
    Y_max = max_values[1]
    Z_min = min_values[2]
    Z_max = max_values[2]
    max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0
    
    mid_x = (X_max+X_min) * 0.5
    mid_y = (Y_max+Y_min) * 0.5
    mid_z = (Z_max+Z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, 0)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')
    ax.set_title('Extrinsic Parameters Visualization')
    plt.show()"""

    # animation for rotating plot
    """
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)
    """
