import numpy as np
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from matplotlib import pyplot as plt
import time
from progress.bar import Bar
import json
import random

# Assuming you have an array of images called "data" with shape (60000, H, W)
num_members = 20
member_size = 3000
num_triplets = 40000 # 60000

# load data
# filename = "../../../../DataSSD/Datasets/LBS/lbs2d_skip3_ensemble21.pkl"
filename = "../../../../DataSSD/Datasets/LBS/lbs2d_skip3_ensemble21m_f32.pkl"
with open(filename, 'rb') as pkl_file:
    data = pickle.load(pkl_file)
print(data.shape)
print("data type:", data.dtype)

# Reshape data from (60000, H, W) to (20, 3000, H, W)
# data_reshaped = data.reshape((num_members, -1, *data.shape[1:]))
# print("data_reshaped:", data_reshaped.shape)

# Create an empty array to store the triplets and corresponding t values
triplet_data = np.zeros((num_triplets, 4, *data.shape[1:]), dtype=np.float32) # (20000, 4, 3, 100, 400)
print("triplet_data:", triplet_data.shape)
print("triplet_data type:", triplet_data.dtype)
# input("data")

triplet_idx = 0
with Bar("Loading the data...", max=num_triplets) as bar:
    while triplet_idx < num_triplets:

        # Randomly select triplets
        for i in range(num_triplets):
            # Randomly select a member
            member_idx = np.random.randint(num_members) + 1

            # Randomly select indices for the triplet
            # k, t, l = np.random.choice(range(data_reshaped.shape[1]), size=3, replace=False)

            # t in the middle, random boundaries
            # window = 6
            # t = np.random.randint(window, member_size - window - 1)
            # t = t * member_idx # rand select member
            # rand_w = np.random.randint(2, window + 1)
            # k = t - rand_w
            # l = t + rand_w

            window = 8 # 6 16
            t = np.random.randint(window, member_size - window - 1)
            t = t * member_idx # rand select member
            # Randomly select k and l within the window around t
            k = np.random.randint(max(0, t - window), t)
            l = np.random.randint(t + 1, min(member_idx * member_size, t + window + 1))
            
            # k = t - window
            # l = t + window

            # k, t, l = k * member_idx, t * member_idx, l * member_idx
            print(" ", k, t, l)

            triplet_data[triplet_idx, 0] = data[k]
            triplet_data[triplet_idx, 1] = data[t]
            triplet_data[triplet_idx, 2] = data[l]

            # Calculate t value and create mask
            t_value = (t - k) / (l - k)
            mask = np.zeros((1, 3, 100, 400))
            mask[0] = 1
            triplet_data[triplet_idx, 3] = mask * t_value
            # print("t_mask:", triplet_data[triplet_idx, 3])

            triplet_idx += 1
            bar.next()

print('Triplet data shape:', triplet_data.shape)
print(triplet_data.dtype)
# triplet_data = np.float32(triplet_data)
# print(triplet_data.dtype)
# Save the triplets as a pickle file
# filename = 'lbs2d_triplets_W6middleT_40K_train.pkl'
# filename = 'lbs2d_triplets_W3fix_30K_train.pkl' # there was a bug
filename = 'lbs2d_triplets_W8_40K_train.pkl'
with open(filename, 'wb') as f:
    pickle.dump(triplet_data, f, protocol=4)
print('Saved!')
