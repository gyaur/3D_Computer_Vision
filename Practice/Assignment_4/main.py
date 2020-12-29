import cv2 as cv
import pandas as pd
import sys
import numpy as np


def estimate_fundamnetal_matrix(src_points, dst_points, normalized_src_points,
                                normalized_dst_points, T1, T2, max_iter,
                                threshold):
    best_inlier_count = -1
    best_F = None
    best_inliers = None
    for _ in range(max_iter):
        indices = np.random.choice(len(src_points), 8, replace=False)
        selected_src_points = normalized_src_points[indices]
        selected_dst_points = normalized_dst_points[indices]

        F = LSQ_fundamental_matrix(src_points=selected_src_points,
                                   dst_points=selected_dst_points)
        F = T2.T @ F @ T1

        num_inliers, inlier_inds = num_inlier(src_points=src_points,
                                              dst_points=dst_points,
                                              F=F,
                                              threshold=threshold)

        if num_inliers > best_inlier_count:
            best_inlier_count = num_inliers
            best_F = np.copy(F)
            best_inliers = inlier_inds

    print(best_inlier_count)

    return best_F, best_inliers


def num_inlier(src_points, dst_points, F, threshold):
    inlier_inds = []
    for ind, (src, dst) in enumerate(zip(src_points, dst_points)):
        pt1 = np.array([*src, 1])
        pt2 = np.array([*dst, 1])

        lL = F.T @ pt2
        lR = F @ pt1

        aL, bL, cL = lL.ravel()
        tL = np.abs(aL * src[0] + bL * src[1] + cL)
        dL = np.sqrt(aL * aL + bL * bL)
        distanceL = tL / dL

        aR, bR, cR = lR.ravel()
        tR = np.abs(aR * dst[0] + bR * dst[1] + cR)
        dR = np.sqrt(aR * aR + bR * bR)
        distanceR = tR / dR

        dist = (distanceL + distanceR) / 2

        if dist < threshold:
            inlier_inds.append(ind)

    return len(inlier_inds), inlier_inds


def LSQ_fundamental_matrix(src_points, dst_points) -> np.ndarray:
    num_points = len(src_points)
    A = np.empty((num_points, 9))
    for ind, (src, dst) in enumerate(zip(src_points, dst_points)):

        x1, y1 = src.ravel()
        x2, y2 = dst.ravel()

        A[ind, 0] = x1 * x2
        A[ind, 1] = x2 * y1
        A[ind, 2] = x2
        A[ind, 3] = y2 * x1
        A[ind, 4] = y2 * y1
        A[ind, 5] = y2
        A[ind, 6] = x1
        A[ind, 7] = y1
        A[ind, 8] = 1

    _, _, eigen_vecs = cv.eigen(A.T @ A)

    F = eigen_vecs[8, :].reshape(3, 3)

    return F


def get_projection_matrices(E, K, src_point, dst_point):
    P1 = K @ np.eye(3, 4)
    U, s, V = np.linalg.svd(E)
    D = np.diag(s)

    if np.linalg.det(U) < 0:
        U[:, 2] *= -1
    if np.linalg.det(V) < 0:
        V[2] *= -1

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    rotation1 = U @ W @ V
    rotation2 = U @ W.T @ V

    if np.linalg.det(rotation1) < 0:
        rotation1 *= -1
    if np.linalg.det(rotation2) < 0:
        rotation2 *= -1

    translation = (U[:, 2] / np.linalg.norm(U[:, 2])).reshape(3, 1)

    P21 = K @ np.concatenate((rotation1, translation), axis=1)
    P22 = K @ np.concatenate((rotation2, translation), axis=1)
    P23 = K @ np.concatenate((rotation1, -translation), axis=1)
    P24 = K @ np.concatenate((rotation2, -translation), axis=1)

    def distance(P2):
        projected_point = linear_triangulation(P1=P1,
                                               P2=P2,
                                               src_point=src_point,
                                               dst_point=dst_point)

        proj1 = P1 @ projected_point
        proj2 = P2 @ projected_point

        if proj1[2] < 0 or proj2[2] < 0:
            return np.inf

        proj1 = (proj1 / proj1[2])[:2]
        proj2 = (proj2 / proj2[2])[:2]

        tmp_d1 = proj1 - src_point
        tmp_d2 = proj2 - dst_point

        return np.sqrt(np.dot(tmp_d1, tmp_d1)) + np.sqrt(np.dot(
            tmp_d2, tmp_d2))

    possible_P2 = (P21, P22, P23, P24)
    P2 = min(possible_P2, key=distance)

    return P1, P2


def linear_triangulation(P1, P2, src_point, dst_point):
    A = np.empty((4, 3))
    b = np.empty((4, 1))

    px, py = src_point
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12 = P1.ravel()

    A[0] = np.array([px * p9 - p1, px * p10 - p2, px * p11 - p3])
    A[1] = np.array([py * p9 - p5, py * p10 - p6, py * p11 - p7])
    b[:2] = np.array([p4 - px * p12, p8 - py * p12]).reshape(2, 1)

    px, py = dst_point
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12 = P2.ravel()

    A[2] = np.array([px * p9 - p1, px * p10 - p2, px * p11 - p3])
    A[3] = np.array([py * p9 - p5, py * p10 - p6, py * p11 - p7])
    b[2:] = np.array([p4 - px * p12, p8 - py * p12]).reshape(2, 1)

    # point = np.linalg.inv(A) @ b  # not stable
    # point = np.linalg.inv(A.T @ A) @ A.T @ b
    point = np.linalg.pinv(A) @ b

    out = np.array([*point.ravel(), 1])
    return out


def normalize(src_points, dst_points):
    # pts = np.concatenate((src_points, dst_points), axis=-1)
    normalized_src_points = np.copy(src_points)
    masspoint = np.average(normalized_src_points, axis=0)
    avg_distance = np.average(
        np.linalg.norm(normalized_src_points, axis=1, ord=2))
    ratio = np.sqrt(2) / avg_distance
    normalized_src_points -= masspoint
    normalized_src_points *= ratio
    T1 = np.eye(3)
    T1[0, 0] = ratio
    T1[1, 1] = ratio
    T1[0, 2] = -ratio * masspoint[0]
    T1[1, 2] = -ratio * masspoint[1]

    normalized_dst_points = np.copy(dst_points)
    masspoint = np.average(normalized_dst_points, axis=0)
    avg_distance = np.average(
        np.linalg.norm(normalized_dst_points, axis=1, ord=2))
    ratio = np.sqrt(2) / avg_distance
    normalized_dst_points -= masspoint
    normalized_dst_points *= ratio
    T2 = np.eye(3)
    T2[0, 0] = ratio
    T2[1, 1] = ratio
    T2[0, 2] = -ratio * masspoint[0]
    T2[1, 2] = -ratio * masspoint[1]

    return normalized_src_points, normalized_dst_points, T1, T2


if __name__ == "__main__":
    img1 = cv.imread(sys.argv[1])
    img2 = cv.imread(sys.argv[2])

    points = pd.read_csv(sys.argv[3], sep=" ", header=None)
    points.index = pd.Index(["u1", "v1", "u2", "v2"])
    points = points.transpose()

    K = np.array(pd.read_csv(sys.argv[4], sep=" ", header=None))

    src_points = np.array(points[["u1", "v1"]])
    dst_points = np.array(points[["u2", "v2"]])

    normalized_src_points, normalized_dst_points, T1, T2 = normalize(
        src_points=src_points, dst_points=dst_points)

    F, inlier_inds = estimate_fundamnetal_matrix(
        src_points=src_points,
        dst_points=dst_points,
        normalized_src_points=normalized_src_points,
        normalized_dst_points=normalized_dst_points,
        T1=T1,
        T2=T2,
        max_iter=1000,
        threshold=1)

    E = K.T @ F @ K

    src_inliers = src_points[inlier_inds]
    dst_inliers = dst_points[inlier_inds]

    src_point = src_inliers[0]
    dst_point = dst_inliers[0]

    P1, P2 = get_projection_matrices(E, K, src_point, dst_point)

    with open(f"out{sys.argv[3][-5]}.txt", "w") as f:
        for src, dst in zip(src_inliers, dst_inliers):
            point = np.around(linear_triangulation(P1, P2, src, dst), 5)[:3]
            f.write(f"{' '.join(str(x) for x in point)}\n")
