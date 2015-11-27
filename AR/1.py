import numpy as np
import cv2


video_capture = cv2.VideoCapture(0)

#if video_capture.isOpened():  #  check if we succeeded
#    -1

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
