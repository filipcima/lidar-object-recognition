import os
from struct import *

class Point:
    def __init__(self, x, y, z, intensity):
        self.x = x
        self.y = y
        self.z = z
        self.intensity = intensity

    def __repr__(self):
        return f'<Point x={unpack("f",x )[0]}, y={unpack("f", y)[0]}, z={unpack("f", z)[0]}, intensity={unpack("f", intensity)[0]}>'

points = []
for file in os.listdir('/Users/cimafilip/Downloads/training-1/velodyne'):
    if file.endswith('.bin'):
        points = []
        with open(file, 'rb') as f:
            x = f.read(4)
            y = f.read(4)
            z = f.read(4)
            intensity = f.read(4)
            points.append(Point(unpack('f', x)[0], unpack('f', y)[0], unpack('f', z)[0], unpack('f', intensity)[0]))
            while x != b'' or y != b'' or z != b'' or intensity != b'':
                x = f.read(4)
                y = f.read(4)
                z = f.read(4)
                intensity = f.read(4)
                if x == b'':
                    break
                point = Point(unpack('f', x)[0], unpack('f', y)[0], unpack('f', z)[0], unpack('f', intensity)[0])
                points.append(point)
        with open(file + '.pcd', 'w') as of:
            of.write(f'''VERSION .7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH "{len(points)}"
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS "{len(points)}"
DATA ascii''')
            for point in points:
                of.write(f'{point.x} {point.y} {point.z} {point.intensity}\n')

