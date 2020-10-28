from random import choice
from typing import Any, Callable, Iterable, Optional, Tuple
import cv2 as cv
import numpy as np
import random
import argparse
from dataclasses import dataclass
from typing import Iterable
import csv
from math import sqrt
from numpy.core.arrayprint import printoptions
from numpy.lib.function_base import average, insert
from numpy.lib.shape_base import apply_along_axis

EPSILON = 0.0000001
K_SIZE = 3


def load(fname: str) -> np.ndarray:
    temp = []
    with open(fname) as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        csv_reader = csv.reader(csvfile, dialect=dialect)
        for line in csv_reader:
            temp.append([float(item) for item in line[:3]])

    return dialect, np.array(temp)


def distance(point, v) -> float:
    w = np.abs(np.dot(point, v[:-1]) + v[-1])
    return w / np.linalg.norm(v[:-1], 2)


def LSQ(points: np.ndarray, inc_inliers: np.ndarray, threshold: float,
        **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    masspoint = np.zeros((3, ))
    for inlier in inc_inliers:
        masspoint += inlier

    masspoint /= inc_inliers.shape[0]
    normalized_points = inc_inliers - masspoint
    avg_distance = np.average(np.linalg.norm(normalized_points, axis=1))

    ratio = sqrt(3) / avg_distance

    normalized_points *= ratio

    A = normalized_points

    _, _, eigenvecs = cv.eigen(A.T @ A)
    a, b, c = eigenvecs[2, :]
    d = -(a * masspoint[0] + b * masspoint[1] + c * masspoint[2])
    plane = np.array([a, b, c, d])
    dist = lambda point: distance(point, plane)
    distances_mask = np.apply_along_axis(dist, 1, points) < threshold
    inliers = points[distances_mask]
    return inliers, plane


def ransac(points: np.ndarray,
           threshold: float,
           max_iter: int,
           inc_inliers: Optional[np.ndarray] = None,
           local_optimizer: Optional[Callable] = None):

    best_inliers = np.empty((1, ))
    inliers = np.empty((1, ))

    is_inner = inc_inliers is not None

    # if inner ransac only choose from inliers
    sample_origin = inc_inliers if is_inner else points

    best_plane = np.empty((3, ))

    for num_iter in range(max_iter):
        if num_iter % 10 == 0 and inc_inliers is None:
            print(f"\r{round((num_iter/max_iter)*100,2)}%", end="")
        while True:
            sample = random.choices(sample_origin, k=K_SIZE)
            if np.any(sample[0] != sample[1]) and np.any(
                    sample[2] != sample[1]) and np.any(sample[0] != sample[2]):
                break

        #points
        A = sample[0]
        B = sample[1]
        C = sample[2]

        (a, b, c) = (B - A) * (C - A)

        d = -(a * A[0] + b * A[1] + c * A[2])
        plane = np.array([a, b, c, d])

        dist = lambda point: distance(point, plane)
        distances_mask = np.apply_along_axis(dist, 1, points) < threshold
        inliers = points[distances_mask]
        if len(inliers) > len(best_inliers):

            best_inliers = np.array(inliers[:])
            inliers = np.empty((1, ))
            best_plane = np.array([a, b, c, d])
            if is_inner:
                optimized_inliers, optimized_plane = local_optimizer(
                    points=points,
                    threshold=threshold,
                    max_iter=100,
                    inc_inliers=best_inliers)
                if len(optimized_inliers) > len(best_inliers):
                    best_inliers = np.array(optimized_inliers[:])
                    best_plane = np.array(optimized_plane[:])
                else:
                    print("this happens")
                    print("asd")

    print(f"\r100.00%")

    return best_inliers, best_plane


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, default="room.xyz")
    parser.add_argument("iteration", type=int, default=1000)
    parser.add_argument("optimizer", choices=["LSQ", "RANSAC"], default="LSQ")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    optimizers = {"RANSAC": ransac, "LSQ": LSQ}
    local_optimizer = optimizers[args.optimizer]

    dialect, points = load(args.file)

    points = np.array(points)
    best_inlieres, best_line = ransac(points=points,
                                      threshold=0.1,
                                      max_iter=args.iteration,
                                      local_optimizer=LSQ)

    print(f"inliers found: {len(best_inlieres)}")
    print(f"the plane {best_line}")

    def it(points, inliers):
        for point in points:
            color = [0., 0., 0.]
            if any(np.sum(point == inliers, axis=1) == 3):
                color = [1., 0., 0.]
            yield list(point) + color

    with open(f"ransaced_{args.file[:-3]}txt", "w") as f:
        writer = csv.writer(f, dialect)
        writer.writerows(it(points, best_inlieres))
