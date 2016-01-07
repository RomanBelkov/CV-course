import numpy as np
import cv2


def show(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def vector_cross(v):
    n = len(v)
    return [np.cross(v[i], v[j]) for i in xrange(n) for j in xrange(i + 1, n)]


black = (0, 0, 0)
red = (0, 0, 255)
white = (255, 255, 255)
size = 700

img = np.ndarray((size, size, 3), dtype='uint8')
img[:] = white

x1 = 0
x2 = size

points = [[20, 30, 1], [150, 30, 1], [200, 160, 1], [40, 150, 1]]

lines = vector_cross(points)
intersections = [(x / z, y / z, 1) for x, y, z in vector_cross(lines)]

for y1, y2 in [(-(A * x1 + C) / B, -(A * x2 + C) / B) for (A, B, C) in lines]:
    cv2.line(img, (x1, y1), (x2, y2), black)

for (x, y, _) in intersections:
    cv2.circle(img, (x, y), 3, red, -1)

show(img)
