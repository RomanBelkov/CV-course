import cv2

# loading classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('jpeg.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
faces = face_cascade.detectMultiScale(gray_img, 1.2, 4)
for (x, y, w, h) in faces:
    # draw faces
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # selecting faces' zones
    roi_gray = gray_img[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    # detect eyes
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 6)  # http://stackoverflow.com/a/20805153/2875908

    # draw eyes
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
