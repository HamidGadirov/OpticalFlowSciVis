import os
import pickle
import numpy as np
import yt
import matplotlib.pyplot as plt

# Directory containing the .pkl file
pkl_file = 'nyx3d_skip5_128_log10norm_OmM_012_OmB_00215_h_055_test_params.pkl' # +
pkl_file = 'nyx3d_skip5_128_log10norm_OmM_012_OmB_00235_h_055_test_params.pkl' # +
pkl_file = 'nyx3d_skip5_128_log10norm_OmM_0155_OmB_00215_h_055_test_params.pkl' # +
pkl_file = 'nyx3d_skip5_128_log10norm_OmM_0155_OmB_00235_h_055_test_params.pkl' # +
print(pkl_file)

# Create output directory for the images and transfer function plots
output_dir = 'Renderings'
tf_output_dir = 'Renderings'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(tf_output_dir, exist_ok=True)

density = False  # False for velocity rendering, True for density rendering

# Load the .pkl file data
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

data = data[0] # without params tuple
print(data.shape)

# The data has the shape (160, 4, 128, 128, 128)
# where 160 is the number of samples, index 0 is for density, and 1, 2, 3 are the velocity components
data_len, channels, dim_x, dim_y, dim_z = data.shape

bounds = (0.1, 0.7)
tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
tf.sample_colormap(np.log10(0.2), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
tf.sample_colormap(np.log10(0.3), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
tf.sample_colormap(np.log10(0.4), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
tf.sample_colormap(np.log10(0.5), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")

# Loop over the 160 samples and render the velocity fields
for idx in range(data_len):
    print(f"Processing frame {idx+1}/{data_len}")

    # If rendering density
    if density:
        density_data = data[idx, 0]  # Get the density data (index 0)
        print(f"Density range for frame {idx}: min={np.min(density_data)}, max={np.max(density_data)}")
        density_data = np.log10(density_data)
        density_data = (density_data - np.min(density_data)) / (np.max(density_data) - np.min(density_data))
        print("Density scaled to [0, 1] range!")
    
    # If rendering velocity
    else:
        velocity_x = data[idx, 1]  # Velocity in x-direction
        velocity_y = data[idx, 2]  # Velocity in y-direction
        velocity_z = data[idx, 3]  # Velocity in z-direction

        # Rendering X-velocity
        ds = yt.load_uniform_grid(dict(velocity_x=velocity_x), velocity_x.shape)
        sc = yt.create_scene(ds, field=("stream", "velocity_x"))
        source = sc[0]
        source.set_log(True)
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        tf_vel.add_gaussian(np.log10(0.17), width=0.0005, height=[1.0, 0.50, 0.05, 1e2])  # orange
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(output_dir, f"flow_x_gt_tf_{idx:04d}.png"), profile_field=("stream", "velocity_x"))
        cam = sc.camera
        cam.resolution = 256
        cam.zoom(0.8)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        im.write_png(os.path.join(output_dir, f"Flow_x_{idx:04d}.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print(f"Rendered X-velocity for frame {idx}!")

        # Rendering Y-velocity
        ds = yt.load_uniform_grid(dict(velocity_y=velocity_y), velocity_y.shape)
        sc = yt.create_scene(ds, field=("stream", "velocity_y"))
        source = sc[0]
        source.set_log(True)
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        tf_vel.add_gaussian(np.log10(0.27), width=0.0005, height=[0.1, 0.8, 0.1, 1e2])  # green
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(output_dir, f"flow_y_gt_tf_{idx:04d}.png"), profile_field=("stream", "velocity_y"))
        cam = sc.camera
        cam.resolution = 256
        cam.zoom(0.8)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        im.write_png(os.path.join(output_dir, f"Flow_y_{idx:04d}.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print(f"Rendered Y-velocity for frame {idx}!")

        # Rendering Z-velocity
        ds = yt.load_uniform_grid(dict(velocity_z=velocity_z), velocity_z.shape)
        sc = yt.create_scene(ds, field=("stream", "velocity_z"))
        source = sc[0]
        source.set_log(True)
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        tf_vel.add_gaussian(np.log10(0.37), width=0.0005, height=[0.1, 0.1, 1.0, 1e2])  # blue
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(output_dir, f"flow_z_gt_tf_{idx:04d}.png"), profile_field=("stream", "velocity_z"))
        cam = sc.camera
        cam.resolution = 256
        cam.zoom(0.8)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        im.write_png(os.path.join(output_dir, f"Flow_z_{idx:04d}.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print(f"Rendered Z-velocity for frame {idx}!")

        # Load the images
        img_x = plt.imread(os.path.join(output_dir, f"Flow_x_{idx:04d}.png"))
        img_y = plt.imread(os.path.join(output_dir, f"Flow_y_{idx:04d}.png"))
        img_z = plt.imread(os.path.join(output_dir, f"Flow_z_{idx:04d}.png"))

        # Blend the images
        blended_img = (img_x + img_y + img_z)
        print(f"Data range for blended_img {idx}: min={np.min(blended_img)}, max={np.max(blended_img)}")

        # Clip the values to the range [0, 1] to avoid invalid values for RGB
        # blended_img = np.clip(blended_img, 0, 1)

        # Visualize the blended image
        fig, ax = plt.subplots()
        ax.imshow(blended_img, vmin=0.0, vmax=1.0, cmap='gray')
        ax.axis('off')

        # Save the blended image
        output_img_path = os.path.join(output_dir, f'frame_{idx:04d}.png')
        plt.savefig(output_img_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved blended image for frame {idx} as {output_img_path}!")
