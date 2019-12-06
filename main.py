# Main file for running both parts of the CSC420 Final Project for Visualizing Stereo Pairs.
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import math
import argparse
import subprocess

# Converts colours of the rectified image to new image channels based on formulas from paper. 
# image_name is the name of the image (eg. veronica), and rectified specifies if this is 
# simple parallel compositing or using the rectified images. 
def AnaglyphColoring(image_name, img, other_img, rectified=False):
    # Separate image into channels
    R = img.copy()
    R[:,:,1] = 0
    R[:,:,2] = 0
    
    G = img.copy()
    G[:,:,0] = 0
    G[:,:,2] = 0

    B = img.copy()
    B[:,:,0] = 0
    B[:,:,1] = 0

    result_img = img.copy()
    result_img[:, :, 2] = 0

    # Now with the other image, just get the R colour channel
    other_img_R = other_img.copy()
    other_img_R[:, :, 0] = 0
    other_img_R[:, :, 1] = 0
    other_img_R = other_img_R

    result_img = other_img_R + result_img
    if rectified == True:
        cv2.imwrite("Anaglyphs/" + image_name + "_anaglyph_rectified" + ".jpg", result_img)
    else:
        cv2.imwrite("Anaglyphs/" + image_name + "_anaglyph.jpg", result_img)
    return result_img

# A4 Code with slight modifications. 
# Code taken from: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
def MatchImages(img1, img2, output_filename):
  # Using ORB descriptors instead of SIFT
  # Initiate ORB detector
  orb = cv2.ORB_create()
  # find the keypoints and descriptors with ORB
  kp1, des1 = orb.detectAndCompute(img1,None)
  kp2, des2 = orb.detectAndCompute(img2,None)

  # create BFMatcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  # Match descriptors.
  matches = bf.match(des1,des2)
  # Sort them in the order of their distance.
  matches = sorted(matches, key = lambda x:x.distance)

  eight_matches = matches[:16]
  angle_list = []
  pruned_matches = []
  # Weed out the bad matches
  for match in matches:
    x1 = kp1[match.queryIdx].pt[0]
    y1 = kp1[match.queryIdx].pt[1]
    x2 = kp2[match.queryIdx].pt[0]
    y2 = kp2[match.queryIdx].pt[1]
    degrees = math.degrees(math.atan2(y2-y1, x2-x1))
    dist = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
    if dist < 30:
        angle_list.append(dist)
        pruned_matches.append(match)

  # Draw matches
  out_img = cv2.drawMatches(img1,kp1,img2,kp2, eight_matches ,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

  cv2.imwrite(output_filename, out_img)
  return (out_img, eight_matches, kp1, kp2, des1, des2)

def RectifyImage(img1, img2, left_points, right_points, F_matrix):
  # Get extrinsic and intrinsic parameters
  (ex, h1, h2) = cv2.stereoRectifyUncalibrated(left_points, right_points, F_matrix, img1.shape[:2])
  # Get combination transform and warp the image
  comb_trans = np.linalg.inv(h1).dot(h2)
  im_warp = cv2.warpPerspective(img1, comb_trans, (img2.shape[1], img1.shape[0]))
  return im_warp

def OutputRectified(img1, img2, image_name):
    left_points = []
    right_points = []
    matched = MatchImages(img1, img2, "Matched/" + image_name + "_match.jpg")
    kp1 = matched[2]
    kp2 = matched[3]
    for match in matched[1]:
        p1 = kp1[match.queryIdx].pt
        p2 = kp2[match.trainIdx].pt
        left_points.append(p1)
        right_points.append(p2)

    left_points = np.array(left_points)
    right_points = np.array(right_points)

    fundamental_mat_leftright = cv2.findFundamentalMat(left_points, right_points)[0]
    fundamental_mat_rightleft = cv2.findFundamentalMat(right_points, left_points)[0]

    rectified_first = RectifyImage(img1, img2, left_points, right_points, fundamental_mat_leftright)
    rectified_second = RectifyImage(img2, img1, right_points, left_points, fundamental_mat_rightleft)
    plt.imshow(rectified_first)
    cv2.imwrite("Rectified/" + image_name + "_rectified_first.jpg", rectified_first)
    cv2.imwrite("Rectified/" + image_name + "_rectified_second.jpg", rectified_second)

    return (rectified_first, rectified_second)

# Run the Middlebury Stereo Evaluation SDK
def RunMiddlebury(img1, img2, image_name):
    filename = cv2.imread("results/disp0.pfm")
    plt.imshow(filename)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Image name refers to the prefix of the image pair. images are named like:
    # veronica, foot, nike, etc. they are attached to the suffix
    # veronica_left, foot_right, etc. 
    # this program *assumes* there is already a left and right image pair for the given image name.
    # inside of the folder image_pairs/
    parser.add_argument("image_name")
    # Technique refers to the usage of outputting anaglyph images or the disparity for a 3D model.
    parser.add_argument("technique")
    args = parser.parse_args()
    print("Image Name : " + args.image_name)
    image_name = args.image_name
    technique = args.technique

    if technique == "1":
        print("Running Anaglyphs...")
        first_image = cv2.imread("image_pairs/" + image_name + "_left.jpg")
        second_image = cv2.imread("image_pairs/" + image_name + "_right.jpg")
        # Run this normally on the original two images
        original_anaglyph = AnaglyphColoring(image_name, first_image, second_image, False)
        # Now get rectified images
        rectified_pair = OutputRectified(first_image, second_image, image_name)
        rectified_anaglyph = AnaglyphColoring(image_name, rectified_pair[0], rectified_pair[1], True)
        plt.imshow(original_anaglyph)
        plt.show()
        plt.imshow(rectified_anaglyph)
        plt.show()

    elif technique == "2":
        print("Running Disparity Map Generation...")
        first_image = cv2.imread("image_pairs/" + image_name + "_left.jpg", cv2.CV_8UC1)
        second_image = cv2.imread("image_pairs/" + image_name + "_right.jpg", cv2.CV_8UC1)
        RunMiddlebury(first_image, second_image, image_name)

        # Run the middlebury script 

    else:
        print("Error: Please choose 1 for Anaglyphs or 2 for Disparity Map Generation.")
    
    
