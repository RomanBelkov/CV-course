import numpy as numpy
import cv2

red = (0, 0, 255)

marks = [[524, 225], [530, 260], [584, 237], [577, 144]]
wall_picture = [[503, 125], [637, 64], [503, 313], [639, 314]]
(w, h) = (700, 700)

img = cv2.imread('hall405.jpg')

for (x, y) in marks:
    img[y, x] = red

pts1 = numpy.float32(wall_picture)
pts2 = numpy.float32([[0, 0], [w, 0], [0, h], [w, h]])

(M, _) = cv2.findHomography(pts1, pts2)
homo_img = cv2.warpPerspective(img, M, (w, h))

cv2.imshow('img', img)
cv2.imshow('homo_img', homo_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
