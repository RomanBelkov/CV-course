import numpy as np
import cv2

img = cv2.imread('image.jpg')
cv2.imshow('Base image', img)
rows, cols = img.shape[:2]


def process_warp_affine((x1, y1), (x2, y2), (x3, y3), (x4, y4)):
    init_points = np.float32([[x1, y1], [x2, y2], [x1, y2]])
    out_points = np.float32([[x3, y3], [x4, y4], [x3, y4]])
    matrix = cv2.getAffineTransform(init_points, out_points)
    return cv2.warpAffine(img, matrix, (cols, rows))


def process_remap((x1, y1), (x2, y2), (x3, y3), (x4, y4)):
    map_x = np.ndarray((rows, cols), dtype='float32')
    map_y = np.ndarray((rows, cols), dtype='float32')
    init_width       = x2 - x1
    init_height      = y2 - y1
    processed_width  = x4 - x3
    processed_height = y4 - y3
    for i in range(rows):
        for j in range(cols):
            map_x[i, j] = x1 + (j - x3) * (init_width / processed_width)
            map_y[i, j] = y1 + (i - y3) * (init_height / processed_height)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)


cv2.imshow('WA', process_warp_affine((50., 50.), (400., 400.), (50., 470.), (400., 200.)))
cv2.imshow('RMP', process_remap((50., 50.), (400., 400.), (50., 470.), (400., 200.)))

cv2.waitKey(0)
cv2.destroyAllWindows()








