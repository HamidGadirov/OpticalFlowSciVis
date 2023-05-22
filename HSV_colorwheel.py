import numpy as np
import matplotlib.pyplot as plt
import colorsys
import math

fig = plt.figure()
ax = fig.add_subplot(projection='polar')

rho = np.linspace(0, 1, 100)  # Radius of 1, distance from center to outer edge
# phi = np.linspace(0, math.pi * 2., 1000)  # in radians, one full circle
# phi = np.linspace(math.pi, math.pi*3.,1000 ) # in radians, one full circle
phi = np.linspace(0.5* math.pi, math.pi * 2.5, 1000)  # in radians, one full circle

RHO, PHI = np.meshgrid(rho, phi)  # get every combination of rho and phi

h = (PHI - PHI.min()) / (PHI.max() - PHI.min())  # use angle to determine hue, normalized from 0-1
# h = np.flip(h)
s = RHO  # saturation is set as a function of radius
v = np.ones_like(RHO)  # value is constant

# convert the np arrays to lists. This actually speeds up the colorsys call
h, s, v = h.flatten().tolist(), s.flatten().tolist(), v.flatten().tolist()
c = [colorsys.hsv_to_rgb(*x) for x in zip(h, s, v)]
c = np.array(c)

ax.scatter(PHI, RHO, c=c)
_ = ax.axis('off')

plt.show()









