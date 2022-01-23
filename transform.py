
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
from shapely import affinity
import math
from DLD_Utils import DLD_Utils as utl
m = utl()

D = 20
G_X = 40
G_R =1
N = 10
pillar1 = Point((0,0)).buffer(10)
pillar2 = affinity.translate(pillar1, xoff=D+G_X, yoff=(D+G_X*G_R)/N)
pillar3 = affinity.translate(pillar1, yoff=(D+G_X*G_R))
pillar4 = affinity.translate(pillar2, yoff=(D+G_X*G_R))

pillar1s = affinity.skew(pillar1, ys=-math.atan(1/N), origin=(0,0), use_radians=True)
pillar2s = affinity.skew(pillar2, ys=-math.atan(1/N), origin=(0,0), use_radians=True)
pillar3s = affinity.skew(pillar3, ys=-math.atan(1/N), origin=(0,0), use_radians=True)
pillar4s = affinity.skew(pillar4, ys=-math.atan(1/N), origin=(0,0), use_radians=True)

pillar1ss = affinity.scale(pillar1s, xfact=1/(D+G_X), yfact=1/(D+G_X*G_R), zfact=1.0, origin=(0,0))
pillar2ss = affinity.scale(pillar2s, xfact=1/(D+G_X), yfact=1/(D+G_X*G_R), zfact=1.0, origin=(0,0))
pillar3ss = affinity.scale(pillar3s, xfact=1/(D+G_X), yfact=1/(D+G_X*G_R), zfact=1.0, origin=(0,0))
pillar4ss = affinity.scale(pillar4s, xfact=1/(D+G_X), yfact=1/(D+G_X*G_R), zfact=1.0, origin=(0,0))

x,y = pillar1ss.exterior.xy
plt.plot(x,y, 'r')
x,y = pillar2ss.exterior.xy
plt.plot(x,y, 'r')
x,y = pillar3ss.exterior.xy
plt.plot(x,y, 'r')
x,y = pillar4ss.exterior.xy
plt.plot(x,y, 'r')

bnd = np.genfromtxt("boundary.csv", delimiter=",")
bnd[:,0], bnd[:,1] = m.parall2square(bnd[:,0], bnd[:,1], 1/N, D, G_X)

plt.scatter(bnd[:,0], bnd[:,1], s=0.5)

plt.show()


# pp = [Point(p) for p in data.tolist()]
# print(pp)