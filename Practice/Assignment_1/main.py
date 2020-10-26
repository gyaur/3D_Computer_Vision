import cv2 as cv
import numpy as np
import random
import argparse

EPSILON = 0.0000001
K_SIZE = 2
MIN_POINTS_FOR_LINE = 250


def detect_edges(img, threshold1, threshold2):
    return cv.Canny(img, threshold1, threshold2) // 255


def ransac(points: np.ndarray, threshold: float, max_iter: int):
    num_iter = 0
    best_inliers = []
    inliers = []
    lines = []

    best_line = np.ones((3, ))

    while (num_iter < max_iter):
        while True:
            sample = random.choices(points, k=K_SIZE)
            if np.any(sample[0] != sample[1]):
                break

        point1 = sample[0]
        point2 = sample[1]

        v = point2 - point1
        v = v / np.linalg.norm(v, 2)

        a = -v[1]
        b = v[0]
        c = -(a * point1[0] + b * point1[1])

        distance = lambda point: abs(a * point[0] + b * point[1] + c)
        inliers = [point for point in points if distance(point) < threshold]
        if len(inliers) > MIN_POINTS_FOR_LINE:
            lines.append(np.array([a, b, c]))
        if len(inliers) > len(best_inliers):
            best_inliers = inliers[:]
            inliers = []
            best_line = np.array([a, b, c])

        num_iter += 1

    return best_line, lines


def draw_line(line, img, color=(0, 0, 255)):
    p1 = (0, round(-line[2] / (line[1] + EPSILON)))
    p2 = (img.shape[1],
          round((-line[0] * img.shape[1] - line[2]) / (line[1] + EPSILON)))
    return cv.line(img, p1, p2, color, 2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("iteration", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    img = cv.imread(args.image, 1)
    edges = detect_edges(img, 150, 200)
    points = []
    for x in range(edges.shape[1]):
        for y in range(edges.shape[0]):
            if edges[y, x] == 1:
                points.append((x, y))

    points = np.array(points)
    bestline, lines = ransac(points=points,
                             threshold=1.0,
                             max_iter=args.iteration)
    for line in lines:
        draw_line(line, img)
    draw_line(bestline, img, (0, 0, 255))

    cv.imshow("RANSAC", img)
    cv.imwrite(f"ransac_{args.iteration}_{args.image}", img)
    cv.waitKey(0)