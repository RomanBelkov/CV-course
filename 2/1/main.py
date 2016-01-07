import numpy as np
import cv2

img = cv2.imread('image.jpg')
rows, cols = img.shape[:2]


def show(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_warp_affine((x1, y1), (x2, y2), (x3, y3), (x4, y4)):
    init_points = np.float32([[x1, y1], [x2, y2], [x1, y2]])
    out_points = np.float32([[x3, y3], [x4, y4], [x3, y4]])
    matrix = cv2.getAffineTransform(init_points, out_points)
    return cv2.warpAffine(img, matrix, (cols, rows))


def process_remap((x1, y1), (x2, y2), (x3, y3), (x4, y4)):
    map_x = np.ndarray((rows, cols), dtype='float32')
    map_y = np.ndarray((rows, cols), dtype='float32')
    init_width       = float(x2 - x1)
    processed_width  = float(x4 - x3)
    init_height      = float(y2 - y1)
    processed_height = float(y4 - y3)
    for i in xrange(rows):
        for j in xrange(cols):
            map_x[i, j] = x1 + float(j - x3) * (init_width / processed_width)
            map_y[i, j] = y1 + float(i - y3) * (init_height / processed_height)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)


# show(process_warp_affine((0, 0), (200, 200), (100, 200), (230, 230)))
# show(process_remap((0, 0), (200, 200), (100, 200), (230, 230)))

