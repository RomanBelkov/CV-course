import numpy as np
import cv2
import math
import copy


# Helping function for find_squares. Taken from OpenCV samples gallery
def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


# Routine for finding squares on image. Taken from OpenCV samples gallery
def find_squares(contours):
    squares = []
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
        if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in xrange(4)])
            if max_cos < 0.1:
                squares.append(np.float32(cnt))
    return squares

# Order must be top-left, top-right, bottom-right, bottom-left, corresponding to the "object_points"
def orient_square2(s):
    s1 = map(tuple, s)
    xs = 1.0 * sum(map(lambda p: p[0], s)) / len(s)
    ys = 1.0 * sum(map(lambda p: p[1], s)) / len(s)
    s1.sort(key=lambda (x, y): math.atan2(y - ys, x - xs))
    return np.float32(map(list, s1))

# draw 3d figure over the found square
def draw_things(rvec, tvec, cam_matrix, dist_coefs):
    COLOR_FRAME  = [0, 255, 0]
    COLOR_MARKER = [0, 255, 255]
    side_w = 0.5
    dx, dy, dz = 0 - (side_w /2.0), 0 - (side_w /2.0), 0 - (side_w /2.0)

    shift_v = lambda v: [v[0] + dx, v[1] + dy, v[2] + dz]

    sides = []
    base = [[0, 0], [side_w, 0], [side_w, side_w], [0.0, side_w]]
    for i in xrange(3):
        for c in [0, side_w]:
            sides.append(map(shift_v, [ v[:i] + [c] + v[i:] for v in base]))

    for i in xrange(len(sides)):
        proj, _ = cv2.projectPoints(np.float32(sides[i]), rvec, tvec, cam_matrix, dist_coefs)
        proj = np.int32(map(lambda x: x[0], proj))
        cv2.polylines(frame, [proj], True, COLOR_FRAME)

    # draw marker on front side of cube
    front_side_marker = map(shift_v, [[0.0, side_w / 2.0, side_w], [side_w, side_w / 2.0, side_w],
                                      [side_w / 2.0, 0.0, side_w], [side_w / 2.0, side_w, side_w]])
    proj, _ = cv2.projectPoints(np.float32(front_side_marker), rvec, tvec, cam_matrix, dist_coefs)
    proj = np.int32(map(lambda x: x[0], proj))
    cv2.polylines(frame, [proj], True, COLOR_MARKER)

# Reading the camera calibration info
calibrated_data = np.load('camera_info.npz')
cam_matrix = calibrated_data['camera_matrix']
dist_coefs = calibrated_data['dist_coefs']

red = (0, 0, 255)

video_capture = cv2.VideoCapture(0)

object_points = np.float32([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]])  # ARE THOSE THE RIGHT VALUES?

while True:
    # Capture frame-by-frame
    _, frame = video_capture.read()
    # To grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Adding a bit of blur
    blurred_gray_image = cv2.blur(gray_frame, (5, 5))
    # Thresholding an image
    _, threshold_image = cv2.threshold(blurred_gray_image, 128.0, 255.0, cv2.THRESH_OTSU)
    # Retrieving countours
    contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Only square contours are interesting to us
    square_contours = find_squares(contours)

    # This beauty allows to calculate rvec & tvec matrixes
    if len(square_contours) > 0:
        # Assume that first square in list is our target
        orien = orient_square2(square_contours[0])
        _, rvec, tvec = cv2.solvePnP(object_points, orien, cam_matrix, dist_coefs)
        draw_things(rvec, tvec, cam_matrix, dist_coefs)

    # Drawing ALL found square contours on image
    cv2.drawContours(frame, np.int32(square_contours), -1, red, 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
