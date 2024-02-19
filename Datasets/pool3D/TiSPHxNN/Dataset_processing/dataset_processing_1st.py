import ti_sph as ts
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def augment(a, steepness = 5, mildrange = 0.2):
    if a < mildrange:
        linear_val = a
        root_val = a**(1/steepness) 
        blend_factor = 3*(a/mildrange)**2 - 2*(a/mildrange)**3
        return linear_val * (1 - blend_factor) + root_val * blend_factor
    else:
        return a**(1/steepness)

def show_attrs_in_path(path, attr):
    import os
    for filename in os.listdir(path):
        if filename.endswith(".npy"):
            data = np.load(path + filename)
            print(filename, data[attr].shape)

def gridExport_density(input_path, output_path, start_index, end_index, vis=False, operations=[]):
    dm_density = ts.Grid_Data_manager(input_path, output_path)
    dm_density.read_data(attr='sensed_density',start_index=start_index,end_index=end_index)
    dm_density.reshape_data_to_2d(index_attr='node_index')
    dm_density.export_single_frame_data('density', operations=operations)

    if vis:
        for i in range(end_index-start_index):
            frame_data = dm_density.processed_data[i]
            frame_data = np.flipud(np.transpose(frame_data))
            normalized_array = ((frame_data - frame_data.min()) * (255 - 0) / (frame_data.max() - frame_data.min())).astype(np.uint8)
            img = Image.fromarray(normalized_array, 'L')
            img.save(f'./output_organised/density_{i}.png')

def gridExport_vel(input_path, output_path, start_index, end_index, vis=False, operations=[]):
    dm_vel = ts.Grid_Data_manager(input_path, output_path)
    dm_vel.read_data(attr='vel',start_index=start_index,end_index=end_index)
    dm_vel.reshape_data_to_2d(index_attr='node_index')
    dm_vel.export_single_frame_data('velocity', operations=operations)

    if vis:
        g_speed = np.sqrt(np.array(dm_vel.processed_data)[:,:,:,0]**2 + np.array(dm_vel.processed_data)[:,:,:,1]**2)
        g_speed_min = g_speed.min()
        g_speed_max = g_speed.max()
        for i in range(end_index-start_index):
            # Step 0
            # Get the velocity vector at each frame
            v = dm_vel.processed_data[i]
            v_x=np.flipud(np.transpose(v[:,:,0]))
            v_y=np.flipud(np.transpose(v[:,:,1]))
            # Step 1
            # Calculate speed and direction (angle) from x and y components of the velocity
            speed = np.sqrt(v_x**2 + v_y**2)
            angle = (np.arctan2(v_x, v_y) + np.pi) / (2. * np.pi)
            # Step 2
            # Create HSV representation
            hsv = np.zeros((v.shape[0],v.shape[1],3))
            hsv[..., 0] = angle
            hsv[..., 1] = 1.0  # Set saturation to maximum
            # hsv[..., 2] = (speed - speed.min()) / (speed.max() - speed.min())  # SINGLE_FRAME Normalize speed to range [0,1]
            hsv[..., 2] = (speed - g_speed_min) / (g_speed_max - g_speed_min)  # GLOBAL Normalize speed to range [0,1]
            # Step 3
            # Convert HSV to RGB
            rgb = colors.hsv_to_rgb(hsv)
            plt.imsave(f'./output_organised/vel_hsv_{i}.png', rgb)
            plt.close()             # if from_zero, then i
    
    return dm_vel
    

def gridExport_strainRate(input_path, output_path, start_index, end_index, operations=[]):
    dm_strainRate = ts.Grid_Data_manager(input_path, output_path)
    dm_strainRate.read_data(attr='strainRate',start_index=start_index,end_index=end_index)
    dm_strainRate.reshape_data_to_2d(index_attr='node_index')
    dm_strainRate.export_single_frame_data('strainRate', operations=operations)


