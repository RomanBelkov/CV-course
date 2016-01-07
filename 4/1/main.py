import numpy as np
import numpy.linalg as la
import numpy.random as random


def normalize(v):
    norm = la.norm(v)
    if norm == 0.0:
        return v
    return v / norm


def plane(points):
    count = len(points)
    right_part = np.array([1.0 for _ in xrange(0, count)])
    mat_a = points.copy()
    x = (la.inv(mat_a.T.dot(mat_a))).dot(mat_a.T).dot(right_part)
    return normalize(x)


o = [0.0, 0.5, 3.0]
v1 = normalize(np.array([3.0, 3, 3]))
v2 = normalize(np.array([3.0, -3, 3]))
v3 = normalize(np.cross(v1, v2))

pts = np.array([o + random.uniform() * v1 + random.uniform() * v2 for i in xrange(100)])

deviated_points = pts.copy() + random.normal(0.0, 0.1, [len(pts), 3])

f = open('out.txt', 'w')
for i in xrange(10, 101, 10):
    res_vector = plane(deviated_points[:i])
    print res_vector, v3
    f.write(str(res_vector + v3) + "\n")
