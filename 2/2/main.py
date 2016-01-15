import numpy as np
import cv2


def vector_cross(v):
    n = len(v)
    return [np.cross(v[i], v[j]) for i in range(n) for j in range(i + 1, n)]


# colors
black = (0, 0, 0)
red = (0, 0, 255)
white = (255, 255, 255)

x0 = 0
size = 700

# creating white image
img = np.ndarray((size, size, 3), dtype='uint8')
img[:] = white

points = [[40, 30, 1], [310, 30, 1], [40, 70, 1], [350, 80, 1]]

lines = vector_cross(points)
intersections = [(x / z, y / z, 1) for x, y, z in vector_cross(lines)]

# drawing lines
for i in xrange(len(lines)):
    A, B, C = lines[i]
    if B != 0:
        y1, y2 = -C / B, -(A * size + C) / B
        cv2.line(img, (0, y1), (size, y2), black)
    else:
        ca = -C / A
        cv2.line(img, (ca, 0), (ca, size), black)

for (x, y, _) in intersections:
    cv2.circle(img, (x, y), 3, red, -1)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
