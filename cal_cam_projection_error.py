"""
Description:
Use to calculate re-projection error, row difference, chessboard distance of stereo camera. This script will store a csv which contains results.
Input path: Images files which store as path/left/*.png and path/right/*.png
Calibration file
Output storage path.
"""

import copy
import os
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import math

# Global para here.
corners_horizontal = 11
corners_vertical = 8
corner_distance = 0.06  # in meters
pattern_size = (corners_horizontal, corners_vertical)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.002)
criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
w = pattern_size[1]
h = pattern_size[0]
objp = np.zeros((corners_vertical * corners_horizontal, 3), np.float32)
objp[:, :2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2) * corner_distance

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

use_distance_board = False  # If distance board is used in test, make it True.
distance_board_number = 1
debug = False
useGammaTransform = False  # Turn on gamma transfer on orgin images.
gamma = 0.5   # Gamma setting.
saveHeatImage = True      # Save row difference heat image
rowDiffLimit = 0.5  # Setup corner point row difference upper limit. If row difference is larger than limit, heatmap will make it saturated.


def grab_raw_file(src_dir, suffix='.png'):
    files_list = []
    if src_dir[-1] != '/': src_dir = src_dir + '/'
    for file in os.listdir(src_dir):
        if file.endswith(suffix):
            files_list.append(file)
    files_list.sort()
    return files_list


def getChessboardReprojectionError(img, cam_M1, cam_D1):
    """
    Calculate camera reprojection error from known camera instrinc.
    input: rgb img, camera instrinc matrix, camera distortion parameters.
    output: rms of reporjection error
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if useGammaTransform:
        gray = gammaTransform(gray, gamma)
    # print("find cor...")
    ret_l, corners_l = cv2.findChessboardCorners(gray, pattern_size,
                                                 cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret_l:
        rt = cv2.cornerSubPix(gray, corners_l, (11, 11), (-1, -1), criteria)
        if corners_l[0][0][0] + corners_l[0][0][1] > corners_l[-1][0][0] + corners_l[-1][0][1]:
            corners_l = corners_l[::-1]
        rt, r1, t1 = cv2.solvePnP(objp, corners_l, cam_M1, cam_D1, flags=cv2.SOLVEPNP_ITERATIVE)
        imgpoints_l_rp, _ = cv2.projectPoints(objp, r1, t1, cam_M1, cam_D1)
        error = cv2.norm(corners_l, imgpoints_l_rp, cv2.NORM_L2)
        rms = np.sqrt(np.square(error) / len(imgpoints_l_rp))
        return rms
    else:
        return None



def rectifyStereo(img_l, img_r, cam_M1_l, cam_D1_l, cam_M1_r, cam_D1_r, R_l, R_r, P_l, P_r):
    """
    Get rectify image.
    """
    remap_interpolation = cv2.INTER_LINEAR
    # remap_interpolation = cv2.INTER_NEAREST
    img_shape = img_l.shape[:-1][::-1]
    lmapx, lmapy = cv2.initUndistortRectifyMap(cam_M1_l, cam_D1_l, R_l, P_l, img_shape, cv2.CV_32FC1)
    rmapx, rmapy = cv2.initUndistortRectifyMap(cam_M1_r, cam_D1_r, R_r, P_r, img_shape, cv2.CV_32FC1)
    imgLeftRectified = cv2.remap(img_l, lmapx, lmapy, interpolation=remap_interpolation)
    imgRightRectified = cv2.remap(img_r, rmapx, rmapy, interpolation=remap_interpolation)
    return imgLeftRectified, imgRightRectified


def getRowDiff(img_rect_l, img_rect_r, Q):
    """
    Calculated every corner point row difference. If can't find chessboard or chessboard corners are not equal in left and right image, return None.
    input: rectified image of left and right, Q matrix from calibration file.
    output:
        mean of all corner's row diff
        max of all corner's row diff
        min of all corner's row diff
        avgDistance of chessboard
        heat image of corner's row diff
    """
    b = 1 / Q[3][2]
    f = Q[2][3]
    gray_l = cv2.cvtColor(img_rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_rect_r, cv2.COLOR_BGR2GRAY)
    if useGammaTransform:
        gray_l = gammaTransform(gray_l, gamma)
        gray_r = gammaTransform(gray_r, gamma)
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret_l and ret_r:
        rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        RowDifflist = []
        avgDistance = 0
        map = np.zeros_like(gray_l)
        for i in range(len(corners_l)):
            RowDiff = np.abs(corners_l[i][0][1] - corners_r[i][0][1])
            RowDifflist.append([RowDiff])
            avgDistance += b * f / abs((corners_l[i][0][0] - corners_r[i][0][0]))
            u, v = corners_l[i][0][0], corners_l[i][0][1]
            val = min(255, RowDiff * (255 / rowDiffLimit))
            attention(u, v, val, map)
        RowDiff = np.array(RowDifflist)
        avgDistance = avgDistance / len(corners_l)
        # Generate Row diff heatimage
        heat_img = cv2.applyColorMap(map, cv2.COLORMAP_JET)
        heat_img[(heat_img[:, :, 0] == 128) & (heat_img[:, :, 1] == 0) & (heat_img[:, :, 2] == 0)] = [0, 0, 0]
        return RowDiff.mean(), RowDiff.max(), RowDiff.min(), avgDistance, heat_img
    else:
        return None, None, None, None, None


def attention(u, v, val, map, r=20):
    """
    Mark a single point visual.
    input: u,v :image coordinate
            val: 0~255
            map: figure map
            r:radius
    """
    shape = map.shape
    h, w = shape[0], shape[1]
    intensity = np.linspace(val, 0, r, dtype=np.uint8)
    u, v = int(u), int(v)
    for x in range(max(0, u - r), min(w, u + r)):
        for y in range(max(0, v - r), min(h, v + r)):
            distance = math.ceil(math.sqrt(pow(x - u, 2) + pow(y - v, 2)))

            if distance < r:
                if map[y][x] == 0:
                    map[y][x] = intensity[distance]
                else:
                    map[y][x] = max(map[y][x], intensity[distance])


def gammaTransform(imgGray, gamma):
    """
    Gamma transform on Gray image.
    """
    imgGrayNorm = imgGray / 255
    gammaNorm = np.power(imgGrayNorm, gamma)
    imgGamma = (gammaNorm * 255).astype(np.uint8)
    return imgGamma


def getDistanceBoardCorner(img_rect):
    """
    Find the corner of distance board.

    """
    gray = cv2.cvtColor(img_rect, cv2.COLOR_BGR2GRAY)
    if useGammaTransform:
        gray = gammaTransform(gray, gamma)
    filter_gray = cv2.GaussianBlur(gray, (9, 9), 5)
    circles = cv2.HoughCircles(filter_gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=50, minRadius=5,
                               maxRadius=100)
    if circles is not None:
        if len(circles) > distance_board_number:
            print("Find circles more than expected.")
            return None
        else:
            circles = np.uint16(np.around(circles))  # [x, y, radius]
            print(circles)
            if debug:
                image_copy = img_rect.copy()
                for circle in circles[0, :]:
                    cv2.circle(image_copy, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
                    cv2.circle(image_copy, (circle[0], circle[1]), 2, (255, 0, 0), 2)

            roi_regions = []
            for circle in circles[0, :]:
                distance_board_lt = [int(circle[0] - circle[2]), int(circle[1] + circle[2] * 2.5)]  # x,y
                distance_board_rb = [int(circle[0] + circle[2]), int(circle[1] + 4.5 * circle[2])]  # x,y
                roi_regions.append(
                    [distance_board_lt, distance_board_rb])  # left_top_point, right_bottom_point in X,Y order
            corners = []
            for roi in roi_regions:
                img_roi = gray[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
                corner = cv2.goodFeaturesToTrack(img_roi, maxCorners=1, qualityLevel=0.5,
                                                 minDistance=10)  # Shi-Tomasi corner check method
                if corner is not None:
                    rt = cv2.cornerSubPix(img_roi, corner, (11, 11), (-1, -1), criteria)  # refines corner location
                    corners.append([roi[0][0] + corner[0, 0, 0],
                                    roi[0][1] + corner[0, 0, 1]])  # Add roi offset back, in x,y order.
                    if debug:
                        cv2.circle(image_copy, (roi[0][0] + int(corner[0, 0, 0]),
                                                roi[0][1] + int(corner[0, 0, 1])), 2, (0, 255, 0), 2)
                        cv2.imshow("find_corner", image_copy)
                        cv2.waitKey()
                        cv2.destroyAllWindows()
                    return corners
                else:
                    print("Can't find distance board corner. ")
                    return None
    else:
        print("Can't find any circles. ")
        return None


def getDistanceBoardDistance(corners_l, corners_r, Q):
    """
    Calculate distance board distance.
    """
    b = 1 / Q[3][2]
    f = Q[2][3]
    if len(corners_l) != len(corners_r):
        print("Distances boards detected are not equal in left and right camera. Drop. ")
        print(corners_l)
        print(corners_r)
        return None
    else:
        distances = []
        print("Corners count is : %d" % (len(corners_l)))
        for i in range(len(corners_l)):
            distance = b * f / abs((corners_l[i][0] - corners_r[i][0]))
            distances.append(distance)
    return np.array(distances)


if __name__ == '__main__':
    # Global para
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputImageRoot", type=str, default='./save',
                        help="Input Image root contains left and right image. ")
    parser.add_argument("-c", "--calibrationFile", type=str, default='./save/calib.yml',
                        help="Calibration file to grab Q matrix. ")
    parser.add_argument("-s", "--saveRoot", type=str, default='./save/result',
                        help="Save root place. ")
    args = parser.parse_args()
    imageDataRoot = args.inputImageRoot
    if imageDataRoot[-1] != '/': imageDataRoot = imageDataRoot + '/'
    kd_yml_file = args.calibrationFile
    saveResultRoot = args.saveRoot
    if saveResultRoot[-1] != '/': saveResultRoot = saveResultRoot + '/'
    if not os.path.isdir(saveResultRoot):
        os.makedirs(saveResultRoot)
    savefile = saveResultRoot + "reprojection_error.csv"
    left_subdir = "left"
    right_subdir = "right"

    # Get camera Kp
    ymlfile = cv2.FileStorage(kd_yml_file, cv2.FILE_STORAGE_READ)
    cam_M1_l = ymlfile.getNode('K1').mat()
    cam_D1_l = ymlfile.getNode('D1').mat()
    cam_M1_r = ymlfile.getNode('K2').mat()
    cam_D1_r = ymlfile.getNode('D2').mat()
    R = ymlfile.getNode('R').mat()
    T = ymlfile.getNode('T').mat()
    R_l = ymlfile.getNode('R1').mat()
    R_r = ymlfile.getNode('R2').mat()
    P_l = ymlfile.getNode('P1').mat()
    P_r = ymlfile.getNode('P2').mat()
    Q = ymlfile.getNode('Q').mat()
    ymlfile.release()

    left_files_list = grab_raw_file(imageDataRoot + left_subdir)
    right_files_list = grab_raw_file(imageDataRoot + right_subdir)
    print(left_files_list)
    print(right_files_list)
    left_rp_error, right_rp_error, stereo_rms_error = [], [], []
    row_diff_list, row_diff_max_list, row_diff_min_list = [], [], []
    distance_list, board_distances_list = [], []
    left_corner_x_list, left_corner_y_list, right_corner_x_list, right_corner_y_list = [], [], [], []
    files_index = copy.deepcopy(left_files_list)
    print("Cal camera reprojection errors.")
    for i in tqdm(range(len(left_files_list))):
        print(left_files_list[i])
        image_l = cv2.imread(imageDataRoot + left_subdir + '/' + left_files_list[i], cv2.IMREAD_UNCHANGED)
        image_r = cv2.imread(imageDataRoot + right_subdir + '/' + right_files_list[i], cv2.IMREAD_UNCHANGED)
        error_l = getChessboardReprojectionError(image_l, cam_M1_l, cam_D1_l)
        error_r = getChessboardReprojectionError(image_r, cam_M1_r, cam_D1_r)
        img_rect_l, img_rect_r = rectifyStereo(image_l, image_r, cam_M1_l, cam_D1_l, cam_M1_r, cam_D1_r, R_l, R_r, P_l,
                                               P_r)
        Rowdiff, Rowdiff_max, Rowdiff_min, avgDistance, heatimage = getRowDiff(img_rect_l, img_rect_r, Q)
        if error_l and error_r and Rowdiff and avgDistance is not None:
            left_rp_error.append(error_l)
            right_rp_error.append(error_r)
            row_diff_list.append(Rowdiff)
            row_diff_max_list.append(Rowdiff_max)
            row_diff_min_list.append(Rowdiff_min)
            distance_list.append(avgDistance)
            if saveHeatImage:
                rowHeatimg = cv2.addWeighted(img_rect_l, 0.3, heatimage, 0.7, 0)
                cv2.imwrite(saveResultRoot + "heat_" + left_files_list[i], rowHeatimg)
                print( "Heat image saved: " + "heat_" + left_files_list[i])
            # get distance board distance.
            if use_distance_board:
                corners_l = getDistanceBoardCorner(img_rect_l)
                corners_r = getDistanceBoardCorner(img_rect_r)
                if corners_l and corners_r is not None:
                    board_distances = getDistanceBoardDistance(corners_l, corners_r, Q)
                    board_distances_list.append(board_distances)
                    left_corner_x_list.append(corners_l[0][0])
                    left_corner_y_list.append(corners_l[0][1])
                    right_corner_x_list.append(corners_r[0][0])
                    right_corner_y_list.append(corners_r[0][1])
                else:
                    print("Find distance board error. ")
                    left_corner_x_list.append("")
                    left_corner_y_list.append("")
                    right_corner_x_list.append("")
                    right_corner_y_list.append("")
                    board_distances_list.append("")
        else:
            files_index.remove(left_files_list[i])
            print("Cal rowdiff error.")
    dic = {'left_rp_error': left_rp_error, 'right_rp_error': right_rp_error, 'row_avg_diff': row_diff_list,
           'row_diff_max': row_diff_max_list, 'row_diff_min': row_diff_min_list,
           'Chessboard_avg_distance': distance_list}
    df = pd.DataFrame(dic, index=files_index)
    if use_distance_board:
        dic_corner = {'left_corner_x': left_corner_x_list, 'left_corner_y': left_corner_y_list,
                      'right_corner_x': right_corner_x_list, 'right_corner_y': right_corner_y_list}
        df_corner = pd.DataFrame(dic_corner, index=files_index)
        df_board = pd.DataFrame(board_distances_list, columns=["distance_board"], index=files_index)
        df = pd.concat([df, df_board, df_corner], join='outer', axis=1)
    df.to_csv(savefile, index=True, header=True, sep=',', encoding='utf-8')

    fig, ax = plt.subplots(4, 1, figsize=(8, 8), dpi=200)
    ax[0].set_title('Left Camera Re-projection error')
    ax[0].set_ylabel('Error(pixel)')
    ax[0].grid(True)
    ax[0].plot(np.array(left_rp_error), 'd', color='blue', markersize=0.2)
    ax[1].set_title('right Camera Re-projection error')
    ax[1].plot(np.array(right_rp_error), 'd', color='green', markersize=0.2)
    ax[1].grid(True)
    ax[1].set_ylabel('Error(pixel)')
    ax[2].set_title('Row avg difference')
    ax[2].plot(np.array(row_diff_list), 'd', color='red', markersize=0.2)
    ax[2].grid(True)
    ax[2].set_ylabel('Error(pixel)')
    ax[3].set_title('Chessboard avg distance')
    ax[3].plot(np.array(distance_list), 'd', color='red', markersize=0.2)
    ax[3].grid(True)
    ax[3].set_ylabel('Distance(m)')
    fig.suptitle('Test pic path: {}'.format(imageDataRoot), fontweight="bold")
    plt.subplots_adjust(top=0.87, bottom=0.08, left=0.10, right=0.95, hspace=0.60,
                        wspace=0.35)
    # plt.show()
    fig.savefig(saveResultRoot + "result.png")
