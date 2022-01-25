from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from math import atan


D = 20
N = 10
G_X = 40
G_R = 1


pillar1 = Point((0, 0)).buffer(D/2)
x,y = pillar1.exterior.xy
plt.plot(x,y, 'k')
x,y = pillar1.buffer(10).exterior.xy
plt.plot(x,y, 'r')
plt.show()


print(pillar1.boundary)