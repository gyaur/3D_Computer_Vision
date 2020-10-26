from typing import Tuple
import cv2
import numpy as np
from dataclasses import dataclass

HEIGHT = 500
WIDTH = 500
CIRCLE_COLOR = (255, 255, 0)


@dataclass
class Direction:
    x: int
    y: int

    def __iter__(self):
        yield self.x
        yield self.y

    def __add__(self, other):
        if isinstance(other, Direction):
            self.x += other.x
            self.y += other.y

    def invert_x(self):
        self.x *= -1

    def invert_y(self):
        self.y *= -1


@dataclass
class Position:
    x: int
    y: int

    def __iter__(self):
        yield self.x
        yield self.y

    def __add__(self, other):
        if isinstance(other, Direction):
            self.x += other.x
            self.y += other.y


@dataclass
class Racket:
    start_position: Position
    end_position: Position
    direction: Direction
    color: Tuple[int, int, int] = (0, 255, 0)  #green
    _update: bool = True

    def update_pos(self):
        if self._is_valid_position() and self._update:
            self.end_position + self.direction
            self.start_position + self.direction

    def reset_direction(self):
        self.direction = Direction(0, 0)
        self._update = True

    def update(self, screen):
        cv2.rectangle(screen, (*self.start_position, ), (*self.end_position, ),
                      self.color, -1)

    def _is_valid_position(self) -> bool:
        new_end_pos = self.end_position.x + self.direction.x
        new_start_pos = self.start_position.x + self.direction.x
        return new_end_pos < WIDTH and new_start_pos > 0


@dataclass
class Circle:
    position: Position
    direction: Direction
    color: Tuple[int, int, int] = (255, 255, 0)  #cyan
    radius: int = 20

    def update_pos(self):
        self.position + self.direction

    def update(self, screen):
        cv2.circle(screen, (*self.position, ), self.radius, self.color, -1)

    def intersects(self, racket: Racket) -> bool:
        x_collision = (racket.start_position.x <= self.position.x + self.radius
                       <= racket.end_position.x) or (
                           racket.start_position.x <= self.position.x -
                           self.radius <= racket.end_position.x)

        y_collision = (racket.start_position.y <= self.position.y + self.radius
                       <= racket.end_position.y) or (
                           racket.start_position.y <= self.position.y -
                           self.radius <= racket.end_position.y)

        return x_collision and y_collision

    def hit_side_wall(self) -> bool:
        return (0 >= self.position.x + self.radius
                or self.position.x + self.radius >= WIDTH) or (
                    0 >= self.position.x - self.radius
                    or self.position.x - self.radius >= WIDTH)

    def hit_top_wall(self) -> bool:
        return (0 >= self.position.y - self.radius)

    def is_out_of_bounds(self):
        return self.position.y > HEIGHT

    def bounce(self, racket: Racket):
        if self.intersects(racket):
            self.direction.invert_y()
            self.direction + racket.direction
        if self.hit_side_wall():
            self.direction.invert_x()
        if self.hit_top_wall():
            self.direction.invert_y()
        if self.is_out_of_bounds():
            return "GAME_OVER"


def mouse_move(event, x, y, flags, param):
    global racket
    if event == cv2.EVENT_MOUSEMOVE:
        racket._update = False
        racket_length = racket.end_position.x - racket.start_position.x
        racket.direction = Direction(
            5 * (x + racket_length // 2 - racket.end_position.x), 0)
        racket.end_position.x = x + racket_length // 2
        racket.start_position.x = x - racket_length // 2


if __name__ == "__main__":
    screen = np.zeros((HEIGHT, WIDTH, 3), dtype='uint8')
    circle = Circle(Position(250, 250), Direction(0, -10))
    racket = Racket(Position(200, 480), Position(300, 490), Direction(0, 0))
    cv2.imshow('That weird bouncing ball game', screen)
    cv2.setMouseCallback('That weird bouncing ball game', mouse_move)
    while True:
        key = cv2.waitKey(40)
        if key == ord('d'):
            racket.direction.x = 10
        elif key == ord('a'):
            racket.direction.x = -10

        cv2.imshow('That weird bouncing ball game', screen)
        screen = np.zeros((HEIGHT, WIDTH, 3), dtype='uint8')

        circle.update_pos()
        racket.update_pos()
        circle.update(screen)
        racket.update(screen)
        result = circle.bounce(racket)
        if result == "GAME_OVER":
            print("DONE")
            break

        racket.reset_direction()

cv2.destroyAllWindows()
