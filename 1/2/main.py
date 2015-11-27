import math

tg = math.tan(math.radians(30))
focalLength = 300 / tg

Ax = focalLength * (-1.0 / 6.0) + 300
Ay = focalLength * (1.0 / 6.0) + 300

print (int(Ax), int(Ay))
