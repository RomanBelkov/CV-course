import cv2
import numpy as np
import math


def normalize(x):
    a, b, c = x.item(0, 0), x.item(1, 0), x.item(2, 0)

    if c != 0:
        return int(a / c), int(b / c)
    else:
        return int(a), int(b)


half_side = 10

k = np.matrix(((100, 0, 300, 0), (0, 100, 300, 0), (0, 0, 1, 0)))

a = math.pi / 4 + math.pi / 2

t = np.matrix(
        ((1, 0, 0, half_side),
         (0, 1, 0, half_side),
         (0, 0, 1, -half_side * math.sqrt(8.0 / 3.0)),
         (0, 0, 0, 1)))  # camera coords

r = np.matrix(
        ((1, 0, 0, 0),
         (0, math.cos(a), -math.sin(a), 0),
         (0, math.sin(a), math.cos(a), 0),
         (0, 0, 0, 1)))

rt = r * t
p = k * rt

img1 = np.zeros((600, 600), np.uint8)
img1.fill(255)

coord = [(-half_side, -half_side, 0, 1), (half_side, -half_side, 0, 1), (half_side, half_side, 0, 1),
         (-half_side, half_side, 0, 1), (-half_side, -half_side, 0, 1)]

for i in xrange(len(coord) - 1):
    p1, p2 = normalize(p * np.transpose(np.matrix(coord[i]))), normalize(p * np.transpose(np.matrix(coord[i + 1])))
    cv2.line(img1, p1, p2, (0, 0, 0), 2)

img2 = np.zeros((600, 600), np.uint8)
img2.fill(255)

q = np.matrix(((10, 0, 0, 300), (0, 10, 0, 300), (0, 0, 0, 0))) * rt

for i in xrange(len(coord) - 1):
    p1, p2 = normalize(q * np.transpose(np.matrix(coord[i]))), normalize(q * np.transpose(np.matrix(coord[i + 1])))
    cv2.line(img2, p1, p2, (0, 0, 0), 2)

cv2.imshow('Perspective', img1)
cv2.imshow('Orthographic', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
