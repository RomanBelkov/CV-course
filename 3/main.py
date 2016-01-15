import numpy as np
import cv2


def is_red(p):
    return p[0] < 60 and p[1] < 60 and p[2] > 150


def find_red_area(point, tol):
    red_points = []
    for i in range(int(point[1]) - tol, int(point[1]) + tol):
        for j in range(int(point[0]) - tol, int(point[0]) + tol):
            if is_red(homo_img[i, j]):
                red_points.append([j, i])
    return red_points


red = (0, 0, 255)

marks = [[524, 225], [530, 260], [584, 237], [577, 144]]
wall_picture = [[503, 125], [637, 64], [503, 313], [639, 314]]
(w, h) = (700, 700)

img = cv2.imread('hall405.jpg')

for (x, y) in marks:
    img[y, x] = red

pts1 = np.float32(wall_picture)
pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

(M, _) = cv2.findHomography(pts1, pts2)
homo_img = cv2.warpPerspective(img, M, (w, h))

for m in marks:
    homography_mark = np.dot(M, [m[0], m[1], 1])
    painting_mark = homography_mark[0] / homography_mark[2], homography_mark[1] / homography_mark[2]
    print(painting_mark)
    red_pixels = find_red_area(painting_mark, 15)

    red_x, red_y = 0., 0.
    for r in red_pixels:
        red_x += r[0]
        red_y += r[1]

    red_area_mid = red_x / len(red_pixels), red_y / len(red_pixels)
    print(red_area_mid)

cv2.imshow('img', img)
cv2.imshow('homo_img', homo_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
