import sys
sys.path.append('../..')
import os
import yt
import numpy as np
from natsort import natsorted
# from utils import cond_mkdir

dir = './Simulation_OmM_0155_OmB_00215_h_07'
save_dir = dir + '_npy'
# cond_mkdir(save_dir)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
file_names = natsorted(os.listdir(dir))
print(save_dir)
volume_vel = np.zeros((4, 128, 128, 128), dtype=np.float64)
for member_idx, file_name in enumerate(file_names):
    print(file_name)
    ds = yt.load(os.path.join(dir, file_name), hint='AMReX')

    all_data_level0 = ds.covering_grid(level=0, left_edge=[0, 0.0, 0.0], dims=[128, 128, 128])
    # print(all_data_level0['gas', 'density'].shape)
    volume_den = all_data_level0['gas', 'density']
    volume_vel[0] = all_data_level0['gas', 'velocity_x']
    volume_vel[1] = all_data_level0['gas', 'velocity_y']
    volume_vel[2] = all_data_level0['gas', 'velocity_z']
    volume_vel[3] = all_data_level0['gas', 'velocity_magnitude']
    # print(volume_vel.shape)
    # print(volume.dtype)

    file_name = file_name[3:]
    file_name_den = "den_" + file_name
    print(file_name_den)
    # print(volume_den.dtype)
    np.save(os.path.join(save_dir, file_name_den), volume_den.astype(np.float32))
    file_name_vel = "vel_" + file_name
    print(file_name_vel)
    np.save(os.path.join(save_dir, file_name_vel), volume_vel.astype(np.float32))

    # test
    # ds_numpy = yt.load_uniform_grid(dict(density=volume), volume.shape)
    # sc = yt.create_scene(ds_numpy, field=("gas", "density"))
    # source = sc[0]
    #
    # bounds = (1e6, 1e15)
    # tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
    # tf.add_gaussian(np.log10(1e7), width=0.003, height=[1.0, 0.0, 0.0, 1.5e1])
    # tf.add_gaussian(np.log10(1e8), width=0.003, height=[0.0, 1.0, 0.0, 1.5e1])
    # tf.add_gaussian(np.log10(1e9), width=0.003, height=[0.0, 0.0, 1.0, 1.5e1])
    # tf.add_gaussian(np.log10(1e10), width=0.003, height=[1.0, 1.0, 0.0, 1.5e1])
    # tf.add_gaussian(np.log10(1e11), width=0.003, height=[0.0, 1.0, 1.0, 1.5e1])
    # tf.add_gaussian(np.log10(1e12), width=0.003, height=[1.0, 1.0, 0.6, 1.5e1])
    # tf.add_gaussian(np.log10(1e13), width=0.003, height=[0.5, 0.5, 0.5, 1.5e1])
    # tf.add_gaussian(np.log10(1e14), width=0.003, height=[0.7, 0.7, 0.7, 1.5e1])
    # tf.add_gaussian(np.log10(1e15), width=0.003, height=[0.8, 0.8, 0.8, 1.5e1])
    # source.tfh.tf = tf
    # source.tfh.bounds = bounds
    #
    # # source.tfh.plot("transfer_function.png", profile_field=("gas", "density"))
    # cam = sc.camera
    # cam.resolution = 512
    # cam.zoom(0.9)
    # im = sc.render()
    # im.write_png(f'numpy_{file_name}.png', background=[1.0, 1.0, 1.0, 0.0], sigma_clip=2.0)
