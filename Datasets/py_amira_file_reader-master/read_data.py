from optparse import Values
from tkinter import X
from xml.sax import default_parser_list
# import xarray as xr
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from progress.bar import Bar
import pickle
import re
import pandas as pd

def getListOfFiles(dirName):
    # For the given path, get the List of all files in the directory tree 

    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles  

def visualize_series(data_to_vis, dir_res="Results", title="Data", show=True, save=False):
    fig=plt.figure()
    columns = 5
    rows = 8
    for i in range(1, columns*rows+1 ):
        if (i == data_to_vis.shape[0]):
            break
        img = data_to_vis[i]
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(img, cmap='viridis')
        
    fig = plt.gcf()
    plt.suptitle(title) 
    fig.set_size_inches(12, 9)
    if show:
        plt.show()  
    if save:
        title += ".pdf"
        fig.savefig(os.path.join(dir_res, title), dpi = 200)

filename = 'drop3D/funs00280'

# data = open(filename, encoding= 'unicode_escape')
# data = data.read()
# print(len(data))
"""
dim_size = 256
data = np.zeros((0, dim_size, dim_size, dim_size))

dirName = 'drop3D'

# Get the list of all files in directory tree at given path
listOfFiles = getListOfFiles(dirName)
listOfFiles.sort()
print("listOfFiles is sorted!")
for k in range(30):
    print(listOfFiles[k])
# input("waiting...")

count = 0
maxNumImages = 20
with Bar("Loading the data...", max=maxNumImages) as bar:
    for elem in listOfFiles: 
        if re.search("(10[0-9]|11[0-9])$", elem):
            print(elem)
            tmp_data = np.fromfile(elem, dtype='uint8')
            tmp_data.resize(1, dim_size, dim_size, dim_size)
            data = np.append(data, tmp_data, axis=0)
            count += 1
            if (count==maxNumImages): # load a subset of the dataset
                break
            bar.next()
        
print(data.shape)
input("")

pkl_file = open("drop3D_data.pkl", 'wb')
pickle.dump(data, pkl_file)
pkl_file.close

pkl_file = open("drop3D_data.pkl", 'rb')
data = []
data = pickle.load(pkl_file)
pkl_file.close

print(data.shape)

x = np.squeeze(data[0,:,0,0])
y = np.squeeze(data[0,0,:,0])
z = np.squeeze(data[0,0,0,:])

# from mayavi import mlab
# s = mlab.mesh(x, y, z)
# mlab.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z, label='drop')
plt.show()
"""

# filename = 'pipedcylinder2d.nc'
# Dimensions:  (tdim: 1501, ydim: 150, xdim: 450, const: 1)
# Coordinates:
#   * xdim     (xdim) float32 -0.5 -0.4866 -0.4733 -0.4599 ... 5.473 5.487 5.5
#   * ydim     (ydim) float32 -0.5 -0.4866 -0.4732 -0.4597 ... 1.473 1.487 1.5
#   * tdim     (tdim) float32 0.0 0.01 0.02 0.03 0.04 ... 14.97 14.98 14.99 15.0
# Dimensions without coordinates: const
# Data variables:
#     u        (tdim, ydim, xdim) float32 ...
#     v        (tdim, ydim, xdim) float32 ...
#     nu       (const) float32 ...
#     radius   (const) float32 ...
#     Re       (const) float32 ...
# Regular grid resolution (X x Y x T): 450 x 150 x 1501
# Simulation domain: [-0.5, 5.5] x [-0.5, 1.5] x [0, 15]
# Reynolds Number: 160 
# Kinematic viscosity: 0.00078125
# Obstacles at (0,0) and (3,1) both with radius: 0.0625 

# filename = 'ctbl3d.nc'
# Dimensions:  (zdim: 130, ydim: 384, xdim: 384)
# Coordinates:
#   * xdim     (xdim) float32 0.0 0.02611 0.05222 0.07833 ... 9.948 9.974 10.0
#   * ydim     (ydim) float32 0.0 0.02611 0.05222 0.07833 ... 9.948 9.974 10.0
#   * zdim     (zdim) float32 0.0 0.02481 0.04961 0.07442 ... 3.126 3.15 3.175 3.2
# Data variables:
#     u        (zdim, ydim, xdim) float32 ...
#     v        (zdim, ydim, xdim) float32 ...
#     w        (zdim, ydim, xdim) float32 ...

# data = xr.open_dataset(filename)
# print(data)                         # show all variables inside this dataset
# print(data.temperature.values)      # this is a 180x201x360 numpy array
# print(data.r)                       # radial discretization

# print(data.vx) 
# print(data.vy) 
# print(data.vz) 
# print("u: ", data.u.shape) 
# print("v: ", data.v.shape) 
# print("w: ", data.w.shape) 

# x = data.xdim.values
# print(x) 
# for t in range ()


filename = "../FluidSimML/4000.am" # 'pipedcylinder2d.nc'
print(filename)
# fh = Dataset(filename, mode='r')

# with open(filename, 'r') as f:
#     for line in f:
#         print(line)

# data = pd.read_csv(filename, encoding= 'unicode_escape')

# input("x")

# import py_amira_file_reader.read_amira as read_amira
# data = read_amira.read_amira(filename)
# print(len(data))
# input("x")

data_ = np.fromfile(filename, dtype='uint8')
print(len(data_))
print(data_)
input("x")

with open(filename, 'rb') as f:
  data = f.read()
print(len(data))

input("x")
# xdim = fh.variables['xdim'][:]
# ydim = fh.variables['ydim'][:]
# tdim = fh.variables['tdim'][:]
# tdim_units = fh.variables['tdim'].units
# fh.close()

print(fh['u'])
u = fh['u'][:]
print(u.shape)
# plt.imshow(u[100,:,:]) 
# plt.show()
u = np.asarray(u)
# visualize_series(u[500:], dir_res="Results", title="U", show=True, save=True)

print(fh['v'])
v = fh['v'][:]
print(v.shape)
# plt.imshow(v[100,:,:]) 
# plt.show()
# visualize_series(v[500:], dir_res="Results", title="V", show=True, save=True)

# get magnitude of velocity vectors
uv = []
for i in range(u.shape[0]):
    u_i = u[i]
    v_i = v[i]
    uv.append(np.flip(np.sqrt(u_i*u_i + v_i*v_i), 0))
uv = np.asarray(uv)
visualize_series(uv[500:], dir_res="Results", title="UV", show=True, save=True)

print("uv:", uv.shape)

# TODO:
# 3d datasets
# 2d Fluid Simulation Ensemble for Machine Learning

pkl_filename = "pipedcylinder2d.pkl"

pkl_file = open(pkl_filename, 'wb')
pickle.dump(uv, pkl_file, protocol=4)
pkl_file.close

pkl_file = open(pkl_filename, 'rb')
data = []
data = pickle.load(pkl_file)
pkl_file.close
print(data.shape)

print(fh['w'])
w = fh['w'][:]
print(w.shape)
# plt.imshow(w[100,:,:]) 
# plt.show()
visualize_series(w[500:], dir_res="Results", title="W", show=True, save=True)

# print(type(xdim))
# print(xdim.shape)

# x = np.array(xdim)

# print(type(xdim))
# print(xdim.shape)

# x = data.xdim.values
# # print(type(x))
# # print(x[0].shape)

# # Get some parameters for the Stereographic Projection
# lon_0 = xdim.mean()
# lat_0 = ydim.mean()

# m = Basemap(width=5000000,height=3500000,
#             resolution='l',projection='stere',\
#             lat_ts=10,lat_0=lat_0,lon_0=lon_0)

# # Because our lon and lat variables are 1D,
# # use meshgrid to create 2D arrays
# # Not necessary if coordinates are already in 2D arrays.
# lon, lat = np.meshgrid(xdim, ydim)
# xi, yi = m(lon, lat)

# # Plot Data
# cs = m.pcolor(xi,yi,np.squeeze(tdim))

# Add Grid Lines
# m.drawparallels(np.arange(-80., 81., 10.), labels=[1,0,0,0], fontsize=10)
# m.drawmeridians(np.arange(-180., 181., 10.), labels=[0,0,0,1], fontsize=10)

# Add Coastlines, States, and Country Boundaries
# m.drawcoastlines()
# m.drawstates()
# m.drawcountries()

# Add Colorbar
# cbar = m.colorbar(cs, location='bottom', pad="10%")
# cbar.set_label(tdim_units)

# Add Title
# plt.title('DJF Maximum Temperature')

# plt.show()
