import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import cv2

data = []

pkl_filename = "lbs2d_skip3_1" + ".pkl" 
pkl_file = open(pkl_filename, 'rb')
data_1 = pickle.load(pkl_file)
pkl_file.close
print(data_1.shape)

pkl_filename = "lbs2d_skip3_2" + ".pkl" 
pkl_file = open(pkl_filename, 'rb')
data_2 = pickle.load(pkl_file)
pkl_file.close
print(data_2.shape)

pkl_filename = "lbs2d_skip3_3" + ".pkl" 
pkl_file = open(pkl_filename, 'rb')
data_3 = pickle.load(pkl_file)
pkl_file.close
print(data_3.shape)

pkl_filename = "lbs2d_skip3_c1_t062" + ".pkl" 
pkl_file = open(pkl_filename, 'rb')
data_4 = pickle.load(pkl_file)
pkl_file.close
print(data_4.shape)

pkl_filename = "lbs2d_skip3_c1_t064" + ".pkl" 
pkl_file = open(pkl_filename, 'rb')
data_5 = pickle.load(pkl_file)
pkl_file.close
print(data_5.shape)

data = np.concatenate((data_1, data_2), axis=0)
data = np.concatenate((data, data_3), axis=0)
data = np.concatenate((data, data_4), axis=0)
data = np.concatenate((data, data_5), axis=0)
print(len(data))
pkl_filename = "lbs2d_skip3_combi_cyl_tau" + ".pkl" 
pkl_file = open(pkl_filename, 'wb')
pickle.dump(data, pkl_file, protocol=4)
pkl_file.close
print("Pkl file created")

