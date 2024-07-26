import os
import pickle
import numpy as np

# Define the file paths
pkl_file_path = 'nyx3d_dens_skip5_128_log10norm_OmM_01375_OmB_00225_h_055_STSRINR_train.pkl'
output_dir = 'STSRINR_GT'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the .pkl file
with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)

print(data.shape)

# Convert each 3D volume to raw file
for i, volume in enumerate(data):
    # Flatten the volume in column-major order
    flattened_volume = volume.flatten(order='F')
    
    # Define the file name
    file_name = f'STSR-INR-nyx_Scalar-GT-{i+1:04d}.raw'
    file_path = os.path.join(output_dir, file_name)
    
    # Save the flattened volume to a raw file
    flattened_volume.tofile(file_path)

print(f'Converted {len(data)} volumes to raw files in {output_dir}')
