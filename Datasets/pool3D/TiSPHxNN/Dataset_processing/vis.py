import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image
import matplotlib.colors as colors
import os

def augment(a, steepness = 5, mildrange = 0.2):
    if a < mildrange:
        linear_val = a
        root_val = a**(1/steepness) 
        blend_factor = 3*(a/mildrange)**2 - 2*(a/mildrange)**3
        return linear_val * (1 - blend_factor) + root_val * blend_factor
    else:
        return a**(1/steepness)
augment_vectorized = np.vectorize(augment)


def scivis_R2toR1(input_path, output_path, start_index, end_index, attr_name, stride=1):
    data = []
    for i in range(start_index, end_index):
        data.append(np.load(os.path.join(input_path, f'{attr_name}_{i}.npy')))
    np_data = np.array(data)
    min_data = np_data.min()
    max_data = np_data.max()

    print(min_data, max_data)
    rgb = np.zeros((np_data.shape[1],np_data.shape[2],3))

    for i in range(end_index-start_index):
        if i % stride != 0:
            continue
        val = np_data[i,...]
        rgb.fill(0)

        normalised_val_positive = augment_vectorized(np.clip(val, 0, None) / max_data)
        normalised_val_negative = augment_vectorized(np.clip(val, None, 0) / min_data)
        rgb[:,:,0] = np.where(val > 0, 1, 1 - normalised_val_negative)
        rgb[:,:,1] = 1 - np.where(val > 0, normalised_val_positive, normalised_val_negative)
        rgb[:,:,2] = np.where(val > 0, 1 - normalised_val_positive, 1)

        output_rgb = np.flip(np.transpose(rgb,(1,0,2)),0)
        image.imsave(os.path.join(output_path, f'sci_{attr_name}_{i}.png'), output_rgb)

def scivis_R2toR2(input_path, output_path, start_index, end_index, attr_name, channel_at_end=False):
    data = []
    for i in range(start_index, end_index):
        frame_data = np.load(os.path.join(input_path, f'{attr_name}_{i}.npy'))
        if channel_at_end:
            frame_data = np.transpose(frame_data, (2,0,1))
        data.append(frame_data)
    np_data = np.array(data)

    g_speed = np.sqrt(np_data[:,0,:,:]**2 + np_data[:,1,:,:]**2)
    g_speed_min = g_speed.min()
    g_speed_max = g_speed.max()

    for i in range(end_index-start_index):
        v = np_data[i]
        v_x=np.flipud(np.transpose(v[0,:,:]))
        v_y=np.flipud(np.transpose(v[1,:,:]))
        # Step 1
        # Calculate speed and direction (angle) from x and y components of the velocity
        speed = np.sqrt(v_x**2 + v_y**2)
        angle = (np.arctan2(v_x, v_y) + np.pi) / (2. * np.pi)
        # Step 2
        # Create HSV representation
        hsv = np.zeros((v.shape[1],v.shape[2],3))
        hsv[..., 0] = angle
        hsv[..., 1] = 1.0  # Set saturation to maximum
        # hsv[..., 2] = (speed - speed.min()) / (speed.max() - speed.min())  # SINGLE_FRAME Normalize speed to range [0,1]
        hsv[..., 2] = (speed - g_speed_min) / (g_speed_max - g_speed_min)  # GLOBAL Normalize speed to range [0,1]
        # Step 3
        # Convert HSV to RGB
        rgb = colors.hsv_to_rgb(hsv)
        plt.imsave(os.path.join(output_path, f'sci_{attr_name}_{i}.png'), rgb)
        plt.close()             # if from_zero, then i

def datavis_1darray(input_path, output_path, attr_name, start_index, end_index):
    val_max = float('-inf')
    val_min = float('inf')
    for i in range(start_index, end_index):
        val = np.load(os.path.join(input_path, f'{attr_name}_{i}.npy'))
        val_max = max(val_max, val.max())
        val_min = min(val_min, val.min())

    print(f"datavis_1darray(): max value for {attr_name} is {val_max}")
    print(f"datavis_1darray(): min value for {attr_name} is {val_min}")

    fig, ax = plt.subplots(figsize=(10, 6))
    

    for i in range(start_index, end_index):
        val = np.load(os.path.join(input_path, f'{attr_name}_{i}.npy'))
        x_values = np.arange(len(val))
        y_values = val

        ax.set_title('1D array of Data')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Value')
        ax.grid(True)
        ax.set_ylim(val_min*2, val_max)
        ax.bar(x_values, y_values, edgecolor='black')
        fig.savefig(os.path.join(output_path, f'1darray_{attr_name}_{i}.png'))
        ax.clear()


def datavis_hist(input_path, output_path, attr_name, start_index, end_index):
    fig, ax = plt.subplots(figsize=(10, 6))

    file_paths = [os.path.join(output_path, f'{attr_name}_{i}.png') for i in range(start_index, end_index)]
    for i, file_path in zip(range(start_index, end_index), file_paths):
        hist = np.load(os.path.join(input_path, f'{attr_name}_{i}.npy'))
        x_values = np.arange(len(hist))*(1/128)-0.5
        y_values = hist

        ax.set_yscale('log')
        ax.set_title('Histogram of Data')
        ax.set_xlabel('Vorticity value')
        ax.set_ylabel('Number of nodes')
        ax.grid(True)
        ax.bar(x_values, y_values, width=1/128, edgecolor='black')
        fig.savefig(file_path)
        ax.clear()

def datavis_hist_R2toR1(input_path, output_path, start_index, end_index, attr_name, range_min, range_max, bins=128):
    data = []
    fig, ax = plt.subplots(figsize=(10, 6))
    file_paths = [os.path.join(output_path, f'hist_{attr_name}_{i}.png') for i in range(start_index, end_index)]

    for i, file_path in zip(range(start_index, end_index), file_paths):
        
        data = np.load(os.path.join(input_path, f'{attr_name}_{i}.npy'))
        hist, bins = np.histogram(data, bins=bins, range=(range_min, range_max))

        ax.set_yscale('log')
        ax.set_title('Histogram of Data')
        ax.set_xlabel('Vorticity value')
        ax.set_ylabel('Number of nodes')
        ax.grid(True)
        ax.bar(bins[:-1], hist, width=np.diff(bins), edgecolor='black')

        # Save the figure
        fig.savefig(file_path)
        ax.clear()  # Clear the axes for the next iteration
    print(f"datavis_hist_R2toR1(): Saved {len(file_paths)} files to {output_path}")