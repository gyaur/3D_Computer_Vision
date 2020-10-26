import argparse
import cv2 as cv
import numpy as np
import random

EPSILON = 0.0000001


def detect_edges(img, threshold1, threshold2):
    return cv.Canny(img, threshold1, threshold2) // 255


def ransac(points: np.ndarray, threshold: float, max_iter: int):
    num_iter = 0
    best_inliers = []
    inliers = []

    best_line = np.ones((3, ))
    K_SIZE = 2
    sample = [1.0] * K_SIZE

    while (num_iter < max_iter):
        while True:
            sample = random.choices(points, k=K_SIZE)
            if np.any(sample[0] != sample[1]):
                break

        point1 = np.array(sample[0])
        point2 = np.array(sample[1])

        v = point2 - point1
        v = v / np.linalg.norm(v, 2)

        n = np.array([-v[1], v[0]])

        a = n[0]
        b = n[1]
        c = -(a * point1[0] + b * point1[1])

        distance = lambda point: abs(a * point[0] + b * point[1] + c)
        inliers = [point for point in points if distance(point) < threshold]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers[:]
            inliers = []
            best_line = (a, b, c)

        num_iter += 1

    return best_inliers, best_line


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("iteration", type=int)
    return parser.parse_args()


def draw_line(line, img, color=(0, 0, 255)):
    p1 = (0, round(-line[2] / (line[1] + EPSILON)))
    p2 = (img.shape[1],
          round((-line[0] * img.shape[1] - line[2]) / (line[1] + EPSILON)))
    return cv.line(img, p1, p2, color, 2)


if __name__ == "__main__":
    args = parse_args()
    img = cv.imread(args.image, 1)
    edges = detect_edges(img, 150, 200)
    points = []
    for x in range(edges.shape[1]):
        for y in range(edges.shape[0]):
            if edges[y, x] == 1:
                points.append((x, y))

    print(
        f"Running SeqRANSAC for {args.iteration} iteration on {len(points)} points"
    )

    lines = []
    for n in range(args.iteration):
        print(f"iteration: {n}")
        inliers, line = ransac(points=points, threshold=1.0, max_iter=1000)
        print(f"inliers: {len(inliers)}")
        if len(inliers) < 20:
            print("Not enough inliers")
        else:
            lines.append(line)
            points = [point for point in points if point not in inliers]

        print(f"points remaining: {len(points)}")
    for line in lines:
        draw_line(line, img)
    cv.imshow("RANSSC", img)
    cv.imwrite(f"seqransac_{args.iteration}_{args.image}", img)
    cv.waitKey(0)