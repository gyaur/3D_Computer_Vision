from operator import index
import cv2 as cv
from feature_matcher import get_features
import sys
import numpy as np
from math import sqrt


def calc_homography(src_points, dst_points):

    normalized_src_points, normalized_dst_points, T1, T2 = normalize(
        src_points, dst_points)

    iter_num = len(point_pairs)
    A = np.zeros((2 * iter_num, 9))
    for i, (first, second) in enumerate(
            zip(normalized_src_points, normalized_dst_points)):
        u1 = first[0]
        v1 = first[1]

        u2 = second[0]
        v2 = second[1]

        A[2 * i, 0] = u1
        A[2 * i, 1] = v1
        A[2 * i, 2] = 1.
        A[2 * i, 3] = 0.
        A[2 * i, 4] = 0.
        A[2 * i, 5] = 0.
        A[2 * i, 6] = -u2 * u1
        A[2 * i, 7] = -u2 * v1
        A[2 * i, 8] = -u2

        A[2 * i + 1, 0] = 0.
        A[2 * i + 1, 1] = 0.
        A[2 * i + 1, 2] = 0.
        A[2 * i + 1, 3] = u1
        A[2 * i + 1, 4] = v1
        A[2 * i + 1, 5] = 1.
        A[2 * i + 1, 6] = -v2 * u1
        A[2 * i + 1, 7] = -v2 * v1
        A[2 * i + 1, 8] = -v2

    _, _, eigen_vecs = cv.eigen(A.T @ A)
    # print(eigen_vecs[8, :])
    # print(eigen_vecs)

    H = eigen_vecs[8, :].reshape(3, 3)

    H /= H[-1, -1]

    # print(H)

    return np.linalg.inv(T2) @ H @ T1


def normalize(src_points, dst_points):
    # pts = np.concatenate((src_points, dst_points), axis=-1)
    normalized_src_points = np.copy(src_points[:, :2])
    masspoint = np.sum(normalized_src_points, axis=0)
    masspoint /= normalized_src_points.shape[0]
    avg_distance = np.average(np.linalg.norm(normalized_src_points, axis=1))
    ratio = sqrt(2) / avg_distance
    normalized_src_points -= masspoint
    normalized_src_points *= ratio
    T1 = np.eye(3)
    T1[0, 0] = ratio
    T1[1, 1] = ratio
    T1[0, 2] = -ratio * masspoint[0]
    T1[1, 2] = -ratio * masspoint[1]

    normalized_dst_points = np.copy(dst_points[:, :2])
    masspoint = np.sum(normalized_dst_points, axis=0)
    masspoint /= normalized_dst_points.shape[0]
    avg_distance = np.average(np.linalg.norm(normalized_dst_points, axis=1))
    ratio = sqrt(2) / avg_distance
    normalized_dst_points -= masspoint
    normalized_dst_points *= ratio
    T2 = np.eye(3)
    T2[0, 0] = ratio
    T2[1, 1] = ratio
    T2[0, 2] = -ratio * masspoint[0]
    T2[1, 2] = -ratio * masspoint[1]
    return normalized_src_points, normalized_dst_points, T1, T2


def num_inlier(src_points, dst_points, H, threshold) -> float:
    # print(H.shape,src_points.shape)
    proj_points = (H @ src_points.T).T
    error = np.sqrt(
        np.sum(np.square(dst_points -
                         (proj_points / proj_points[:, -1].reshape(-1, 1))),
               axis=1))
    mask = error < threshold
    return np.sum(mask)


def ransac(src_points, dst_points, threshold, max_iter):
    best_inlier_count = -1
    best_homography = None
    for _ in range(max_iter):
        indices = np.random.choice(len(src_points), len(src_points)//4 , replace=False)
        selected_src_points = src_points[indices]
        selected_dst_points = dst_points[indices]
        H = calc_homography(src_points=selected_src_points,
                            dst_points=selected_dst_points)

        inliers = num_inlier(src_points=selected_src_points,
                             dst_points=selected_dst_points,
                             H=H,
                             threshold=threshold)
        if inliers > best_inlier_count:
            best_inlier_count = inliers
            best_homography = H

    return best_homography, best_inlier_count


def transform_img(img: np.ndarray, new_img: np.ndarray, tr: np.ndarray,
                  is_perspective):

    inv_tr = np.linalg.inv(tr)
    width, height, _ = img.shape
    new_w, new_h, _ = new_img.shape
    img = np.copy(img)

    for x in range(new_w):
        for y in range(new_h):
            pt = np.array([x, y, 1.0])
            ptr = inv_tr @ pt

            if is_perspective:
                ptr = (1.0 / ptr[2]) * ptr

            new_x = round(ptr[0])
            new_y = round(ptr[1])

            if 0 <= new_x < width and 0 <= new_y < height:
                new_img[x, y] = np.copy(img[new_x, new_y])


if __name__ == "__main__":
    img1 = cv.imread(sys.argv[1])
    img2 = cv.imread(sys.argv[2])

    point_pairs = get_features(img1, img2)

    src_points = np.array([(pair[0][1], pair[0][0], 1)
                           for pair in point_pairs])
    dst_points = np.array([(pair[1][1], pair[1][0], 1)
                           for pair in point_pairs])

    H, inlier_count = ransac(src_points=src_points,
                             dst_points=dst_points,
                             threshold=1,
                             max_iter=1000)
    print(inlier_count)

    new_shape = (img1.shape[0] * 2, img1.shape[1] * 2, img1.shape[2])
    new_img = np.zeros(new_shape, dtype=img2.dtype)

    transform_img(img2, new_img, np.eye(3), True)
    transform_img(img1, new_img, H, True)

    cv.namedWindow("Display window")
    cv.imshow("Display window", new_img)
    cv.imwrite(f"{sys.argv[1]}+{sys.argv[2]}.png", new_img)
    cv.waitKey(0)
