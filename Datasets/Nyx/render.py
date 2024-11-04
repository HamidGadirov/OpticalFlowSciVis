import os
import numpy as np
import yt
import matplotlib.pyplot as plt

# Directory containing the .npy files
npy_dir = 'Simulation_OmM_012_OmB_00215_h_055_npy'
# npy_dir = 'Simulation_OmM_012_OmB_00235_h_055_npy'
npy_dir = 'Simulation_OmM_0155_OmB_00215_h_055_npy'
# npy_dir = 'Simulation_col_sep_10_npy'
# npy_dir = 'nyx3d_skip5_128_log10norm_OmM_0155_OmB_00235_h_055_test_params.pkl'

print(npy_dir)

# Create output directory for the images and transfer function plots
output_dir = 'Renderings'
tf_output_dir = 'Renderings'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(tf_output_dir, exist_ok=True)

density = False

# List all .npy files within the range
if density:
    npy_files = sorted([f for f in os.listdir(npy_dir) if f.startswith('den_') and f.endswith('.npy')])
else:
    npy_files = sorted([f for f in os.listdir(npy_dir) if f.startswith('vel_') and f.endswith('.npy')])
npy_files = [f for f in npy_files if int(f.split('_')[-1].split('.')[0]) >= 0 and int(f.split('_')[-1].split('.')[0]) <= 2000]

bounds = (0.1, 0.7)
tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
tf.sample_colormap(np.log10(0.2), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
tf.sample_colormap(np.log10(0.3), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
tf.sample_colormap(np.log10(0.4), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
tf.sample_colormap(np.log10(0.5), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
# velocity
# tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
# tf_vel.sample_colormap(np.log10(0.2), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
# tf_vel.sample_colormap(np.log10(0.3), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
# tf_vel.sample_colormap(np.log10(0.4), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
# tf_vel.sample_colormap(np.log10(0.5), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")

# Loop over files to generate images and save the TF plot for each frame
for idx, file in enumerate(npy_files):
    if idx < 300:
        # print(idx, file)
        continue
    print(idx)

    # Load the numpy data
    file_path = os.path.join(npy_dir, file)
    data = np.load(file_path)
    if density == False:
        
        print("data:", data.shape)
        data = data[:3]
        print("data:", data.shape)
    # print(f"Data range for {file}: min={np.min(data[0])}, max={np.max(data[0])}")

    # vel mag
    # if density == False:
    #     vel_mag = np.sqrt(data[0] ** 2 + data[1] ** 2 + data[2] ** 2)
    #     data = vel_mag
    
    print(f"Data range for {file}: min={np.min(data)}, max={np.max(data)}")

    if density:
        data = np.log10(data)
        print("Density scaled logarithmically!")
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        print("Density scaled to [0, 1] range!")
    else:
        # data = (2 * (data - np.min(data)) / (np.max(data) - np.min(data))) - 1
        for i in range(0, 3):
            data[i] = (2 * (data[i] - np.min(data[i])) / (np.max(data[i]) - np.min(data[i]))) - 1
        print("Velocities scaled to [-1, 1] range!")
        print(f"Data range for {file}: min={np.min(data)}, max={np.max(data)}")

    cam_resolution = 256
    cmap ='gray'

    if density:
        ds = yt.load_uniform_grid(dict(density=data), data.shape)
        # ds = yt.load_uniform_grid(dict(velocity_mag=data), data.shape)

        # Create the rendering object
        sc = yt.create_scene(ds, field="density")
        # sc = yt.create_scene(ds, field=("stream", "velocity_mag"))

        # Let YT generate the default color transfer function automatically
        source = sc[0]
        source.set_log(True)
        source.tfh.tf = tf
        source.tfh.bounds = bounds

        # Render the scene
        cam = sc.camera
        cam.resolution = cam_resolution
        cam.zoom(0.8)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()

        # Save the image for this frame
        output_img_path = os.path.join(output_dir, f'frame_{idx:04d}.png')
        sc.save(output_img_path, sigma_clip=6)
        print(f'Saved rendering for {file} as {output_img_path}')
    else:
        ds = yt.load_uniform_grid(dict(velocity_x=data[0]), data[0].shape)
        sc = yt.create_scene(ds, field=("stream", "velocity_x"))
        source = sc[0]
        source.set_log(True)
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        # tf_vel.add_gaussian(np.log10(0.55), width=2e-4, height=[0.84, 0.15, 0.16, 1e2]) # red
        # tf_vel.add_gaussian(np.log10(0.55), width=3e-4, height=[0.58, 0.40, 0.74, 1e2]) # purple
        tf_vel.add_gaussian(np.log10(0.17), width=0.0005, height=[1.0, 0.50, 0.05, 1e2]) # orange
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(output_dir, "flow_x_gt_tf.png"), profile_field=("stream", "velocity_x"))

        cam = sc.camera
        cam.resolution = cam_resolution
        cam.zoom(0.8)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        im.write_png(os.path.join(output_dir, "Flow_x.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        ds = yt.load_uniform_grid(dict(velocity_y=data[1]), data[1].shape)
        sc = yt.create_scene(ds, field=("stream", "velocity_y"))
        source = sc[0]
        source.set_log(True)
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        # tf_vel.add_gaussian(np.log10(0.55), width=2e-4, height=[0.17, 0.63, 0.17, 1e2]) # green
        tf_vel.add_gaussian(np.log10(0.27), width=0.0005, height=[0.1, 0.8, 0.1, 1e2]) # green
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(output_dir, "flow_y_gt_tf.png"), profile_field=("stream", "velocity_y"))

        cam = sc.camera
        cam.resolution = cam_resolution
        cam.zoom(0.8)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        im.write_png(os.path.join(output_dir, "Flow_y.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        ds = yt.load_uniform_grid(dict(velocity_z=data[2]), data[2].shape)
        sc = yt.create_scene(ds, field=("stream", "velocity_z"))
        source = sc[0]
        source.set_log(True)
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        # tf_vel.add_gaussian(np.log10(0.55), width=2e-4, height=[0.12, 0.47, 0.71, 1e2]) # blue
        tf_vel.add_gaussian(np.log10(0.37), width=0.0005, height=[0.1, 0.1, 1.0, 1e2]) # blue
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(output_dir, "flow_z_gt_tf.png"), profile_field=("stream", "velocity_z"))

        cam = sc.camera
        cam.resolution = cam_resolution
        cam.zoom(0.8)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        im.write_png(os.path.join(output_dir, "Flow_z.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        # Load the images
        img_x = plt.imread(os.path.join(output_dir, "Flow_x.png"))
        img_y = plt.imread(os.path.join(output_dir, "Flow_y.png"))
        img_z = plt.imread(os.path.join(output_dir, "Flow_z.png"))

        # Blend the images
        blended_img = (img_x + img_y + img_z)
        print(f"Data range for blended_img {file}: min={np.min(blended_img)}, max={np.max(blended_img)}")

        # Clip the values to the range [0, 1] to avoid invalid values for RGB
        # blended_img = np.clip(blended_img, 0, 1)

        # Visualize the blended image
        fig, ax = plt.subplots()
        ax.imshow(blended_img, vmin=0.0, vmax=1.0, cmap=cmap)
        ax.axis('off')

        # Save the blended image
        output_img_path = os.path.join(output_dir, f'frame_{idx:04d}.png')
        # plt.imsave(output_img_path, blended_img)
        fig.savefig(output_img_path, dpi = 300)
        print(f'Saved blended image as {output_img_path}')

    # Save the transfer function plot for this frame
    # tf_plot_path = os.path.join(tf_output_dir, f"tf_plot_{idx:04d}.png")
    # source.tfh.plot(tf_plot_path, profile_field=("gas", "density"))
    # print(f'Saved transfer function plot as {tf_plot_path}')


# image_files = natsorted(glob.glob(f'./{save_dir}/.png'))
# first_image = cv2.imread(image_files[0])
# height, width, layers = first_image.shape
# video = cv2.VideoWriter(f'./output.mp4', cv2.VideoWriter_fourcc('mp4v'), 15, (width, height))
# for idx, file in enumerate(image_files):
#     img = cv2.imread(file)
#     img = cv2.putText(img, f'{idx}', (450, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#     video.write(img)
# video.release()
