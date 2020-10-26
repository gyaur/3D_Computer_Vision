import cv2 as cv
import numpy as np
import csv
from math import cos, floor, pi, sin
import sys

u, v, rad = 0.5, 1.0, 100.0
drag = False
prev_x, prev_y = 0, 0


def load(fname: str) -> np.ndarray:
    temp = []
    with open(fname) as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        csv_reader = csv.reader(csvfile, dialect=dialect)
        for line in csv_reader:
            temp.append([float(item) for item in line[:-1]])

    return np.array(temp)


def draw_points(data: np.ndarray, u: float, v: float, rad: float,
                img: np.ndarray):
    num_rows = data.shape[0]

    C = np.zeros((3, 3))
    R = np.zeros((3, 3))

    tx = cos(u) * sin(v)
    ty = sin(u) * sin(v)
    tz = cos(v)

    # C params
    C[0, 0] = 3000.0
    C[0, 2] = 400.0

    C[1, 1] = 3000.0
    C[1, 2] = 300.0

    C[2, 2] = 1.0

    # T params

    T = np.array([rad * tx, rad * ty, rad * tz])

    num_pi = floor(v / pi)

    Z = np.array([-1.0 * tx, -1.0 * ty, -1.0 * tz])
    X = np.array([sin(u) * sin(v), -cos(u) * sin(v), 0.0])

    if num_pi % 2:
        X = (1.0 / np.linalg.norm(X, 2)) * X
    else:
        X = (-1.0 / np.linalg.norm(X, 2)) * X

    up = np.cross(X, Z)

    R[2, 0] = Z[0]
    R[2, 1] = Z[1]
    R[2, 2] = Z[2]

    R[1, 0] = up[0]
    R[1, 1] = up[1]
    R[1, 2] = up[2]

    R[0, 0] = X[0]
    R[0, 1] = X[1]
    R[0, 2] = X[2]

    for i in range(num_rows):
        vec = data[i, :]
        tr_vec = R @ (vec - T)
        tr_vec = C @ tr_vec
        tr_vec = tr_vec / tr_vec[2]

        cv.circle(img, (int(tr_vec[0]), int(tr_vec[1])), 2, (255, 255, 255), 2,
                  0)


def mouse_move(event, x, y, flags, param):
    global rad, drag, u, v, prev_x, prev_y
    if event == cv.EVENT_MOUSEWHEEL:
        if flags > 0:
            rad /= 1.1
        else:
            rad *= 1.1

    if event == cv.EVENT_LBUTTONDOWN:
        drag = True

    if event == cv.EVENT_LBUTTONUP:
        drag = False

    if event == cv.EVENT_MOUSEMOVE:
        dx = x - prev_x
        dy = y - prev_y

        prev_x = x
        prev_y = y

        if drag:
            u += dx * 0.005
            v -= dy * 0.005


if __name__ == "__main__":
    data = load(sys.argv[1])

    screen = np.zeros((600, 800, 3), dtype='uint8')
    draw_points(data, u, v, rad, screen)
    cv.imshow('Cat inspector', screen)
    cv.setMouseCallback('Cat inspector', mouse_move)

    rad = 4000  # set a reasonable zoom level
    auto = True
    time = 0

    while True:
        key = cv.waitKey(16)
        if key == ord('d'):
            u += 0.1
        elif key == ord('a'):
            u -= 0.1
        elif key == ord('w'):
            v += 0.1
        elif key == ord('s'):
            v -= 0.1
        elif key == ord('q'):
            rad *= 1.1
        elif key == ord('e'):
            rad /= 1.1
        elif key == ord('c'):
            auto = not auto

        if auto:
            time += 1
            u = 0.1 * time + 10
            v = 0.1 * time

            scaling_factor = min(2, (sin(time / 10) + 1) + 0.5)  #[0.5,2]
            rad = (scaling_factor * 4000)

        screen = np.zeros((600, 800, 3), dtype='uint8')
        draw_points(data, u, v, rad, screen)
        cv.imshow('Cat inspector', screen)
