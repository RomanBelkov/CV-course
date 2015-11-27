import numpy as np
import cv2

video_capture = cv2.VideoCapture(0)

#if video_capture.isOpened():  #  check if we succeeded
#    -1
red = (0, 0, 255)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred_gray_image = cv2.blur(gray_frame, (5, 5))

    ret, threshold_image = cv2.threshold(blurred_gray_image, 128.0, 255.0, cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(frame, contours, -1, red, 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
