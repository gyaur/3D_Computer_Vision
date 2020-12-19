import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys


def round_pt(pt):
    return (round(pt[0], 2), round(pt[1], 2))

def get_features(img1, img2):

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < (0.75 * n.distance):
            good.append((kp1[m.queryIdx].pt, kp2[m.trainIdx].pt))

    # print(out[:5])
    return good


if __name__ == "__main__":
    img1 = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(sys.argv[2], cv.IMREAD_GRAYSCALE)

    get_features(img1, img2)

# Draw first 10 matches.
# img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()