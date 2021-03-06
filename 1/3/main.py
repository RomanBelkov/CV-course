import cv2
from matplotlib import pyplot as plt

gray_img = cv2.imread('low-contrast.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Bus', gray_img)
height, width = gray_img.shape

hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

# default max/min hist values
maxH = 255
minH = 0
min_pixel_count = 10
r = range(256)

# finding optimal max/min hist borders
for i in r:
    if hist[i] >= min_pixel_count:
        minH = i
        break

for i in reversed(r):
    if hist[i] >= min_pixel_count:
        maxH = i
        break

print(minH, maxH)

processed_img = gray_img

# adjusting image according to modified hist
for x in range(height):
    for y in range(width):
        if minH <= processed_img[x, y] <= maxH:
            processed_img[x, y] = 255 * (processed_img[x, y] - minH) / (maxH - minH)

hist2 = cv2.calcHist([processed_img], [0], None, [256], [0, 256])
cv2.imshow('Processed bus', processed_img)

plt.plot(hist)
plt.plot(hist2)
plt.title('Histograms')
plt.show()
