import numpy as np
import numpy.linalg as la
import numpy.random as random


# normalizes the vector
def normalize(v):
    norm = la.norm(v)
    if norm == 0.0:
        return v
    return v / norm


# planes some points onto the surface
def plane(points):
    count = len(points)
    right_part = np.array([random.uniform(5., 10.) for _ in range(count)])
    mat_a = points.copy()
    x = (la.inv(mat_a.T.dot(mat_a))).dot(mat_a.T).dot(right_part)  # can cheat using la.lstsq here
    return normalize(x)


o = [0.0, 0.5, 0.0]
v1 = normalize(np.array([3.0, 3, 4]))
v2 = normalize(np.array([3.0, -3, -5]))
v3 = normalize(np.cross(v1, v2))

pts = np.array([o + random.uniform() * v1 + random.uniform() * v2 for i in range(100)])

deviation_points = pts.copy() + random.normal(0.0, 0.1, [len(pts), 3])

out_file = open('out.txt', 'w')
for i in range(10, 101, 10):
    res_vector = plane(deviation_points[:i])
    print(res_vector, v3)
    out_file.write(str(res_vector + v3) + "\n")
