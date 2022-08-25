# import xarray as xr
# data = xr.open_dataset('Velocity/ns_1000_v.dat')
# print(data)                         # show all variables inside this dataset
# print(data.temperature.values)      # this is a 180x201x360 numpy array
# print(data.r)                       # radial discretization

#print(data.vx) 
#print(data.vy) 
#print(data.vz) 


import numpy as np

data = np.fromfile('Velocity/ns_1000_v.dat', dtype=np.float32)
# data = np.fromfile('Density/ns_1010_r.dat', dtype=float)
print(data)
print(len(data))

data = np.reshape(data,(128, 128, 128, 3))
print(data.shape)

slice_num = 5 # 100
vx = data[:,:,slice_num,0]
vy = data[:,:,slice_num,1]

print(vx)
print(vy)

print(vx.shape)

import cv2

# Use Hue, Saturation, Value colour model 
hsv = np.zeros((128,128,3), dtype=np.uint8)
# print(img1.shape)
# input("x")
hsv[..., 1] = 255

# print(flow_mat[..., 0])
# flow_mat = flow_mat.astype('float32') 

mag, ang = cv2.cartToPolar(vx, vy)
hsv[..., 0] = ang * 180 / np.pi / 2
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

# print(hsv)

# print(hsv[..., 0])
# hsv = hsv.astype(np.uint8)
# print(cv2.__version__)
# input("x")

bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# print(bgr)
# cv2.imshow("colored flow", bgr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.gca()
plt.imshow(bgr)
plt.show()
# res_path = os.path.join('result', 'result1.png')
fig.savefig('gt_1000_5')

# def is_float(string):
#     """ True if given string is float else False"""
#     try:
#         return float(string)
#     except ValueError:
#         return False

# data = []
# with open('Velocity/ns_1000_v.dat', 'r') as f:
#     d = f.readlines()
#     for i in d:
#         k = i.rstrip().split(",")
#         data.append([float(i) if is_float(i) else i for i in k])

# data = np.array(data, dtype='O')
# print(data)
