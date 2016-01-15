import numpy as np
import numpy.linalg as la
import cv2
from scipy.optimize import leastsq

f = 6741.0
cell_size = 333.0
new_coord = np.array(
        [(231, 1026),
         (1089, 834),
         (899, 5),
         (90, 330),
         (456, 681),
         (740, 616),
         (679, 339),
         (397, 403)], dtype=np.dtype(float))

old_coord = np.array(
        [(0, 0),
         (6 * cell_size, 0),
         (6 * cell_size, 6 * cell_size),
         (0, 5 * cell_size),
         (2 * cell_size, 2 * cell_size),
         (4 * cell_size, 2 * cell_size),
         (4 * cell_size, 4 * cell_size),
         (2 * cell_size, 4 * cell_size)]
)

img = cv2.imread("chessboard.jpg")
k = np.array(
        [[f, 0, img.shape[1] / 2],
         [0, f, img.shape[0] / 2],
         [0, 0, 1]])
hom, _ = cv2.findHomography(old_coord[0:4], new_coord[0:4])
hn = np.dot(la.inv(k), hom)
norm = la.norm(hn[:, 0])
hn = np.divide(hn, norm)
r = hn.copy()
r[:, 2] = np.cross(hn[:, 0], hn[:, 1])

p = np.empty(6)
p[0:3] = cv2.Rodrigues(r)[0].reshape(3)
p[3:6] = hn[:, 2]


def calc_error(res):
    sum = 0.0
    r = cv2.Rodrigues(res[0:3])[0]
    temp = np.zeros((points_num, 3))
    temp[:, :-1] = old_coord[0:points_num]

    for i in range(0, points_num):
        y = np.dot(k, np.add(np.dot(r, temp[i]), hn[:, 2]))
        sum += la.norm(np.array([y[0] / y[2] - new_coord[i][0], y[1] / y[2] - new_coord[i][1]]))

    return sum / points_num


def residuals(x):
    res = np.empty(points_num * 2)
    r = cv2.Rodrigues(x[0:3])[0]
    temp = np.zeros((points_num, 3))
    temp[:, :-1] = old_coord[0:points_num]
    for i in range(0, points_num):
        y = np.dot(k, np.add(np.dot(r, temp[i]), x[3:6]))
        res[2 * i: 2 * i + 2] = np.array([new_coord[i][0] - y[0] / y[2], new_coord[i][1] - y[1] / y[2]])
    return res


output_file = open('output.txt', 'w')
for points_num in range(4, 9):
    print(leastsq(residuals, p)[0])
    output_file.write("%s\n" % calc_error(leastsq(residuals, p)[0]))

output_file.close()
