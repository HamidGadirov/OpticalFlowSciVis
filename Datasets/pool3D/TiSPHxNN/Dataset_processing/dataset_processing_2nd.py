import numpy as np
import ti_sph as ts
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import taichi as ti
import os


def process_strainRate_to(input_path, output_path, start_index, end_index, attr_name='strainRate', to='vorticity', use_density_mask=False):
    data = []
    dm_strainRate = ts.Grid_Data(input_path, attr_name, start_index, end_index)
    dm_density = ts.Grid_Data(input_path, 'density', start_index, end_index)

    ti_vorticity = ti.field(dtype=ti.f32, shape=(dm_strainRate.shape_x-2, dm_strainRate.shape_y-2))

    ti_density_data = ti.field(dtype=ti.f32, shape=(dm_density.shape_x, dm_density.shape_y))
    ti_density_grad_px = ti.field(dtype=ti.f32, shape=(dm_density.shape_x-2, dm_density.shape_y-2))
    ti_density_grad_py = ti.field(dtype=ti.f32, shape=(dm_density.shape_x-2, dm_density.shape_y-2))
    ti_density_grad_map = ti.field(dtype=ti.f32, shape=(dm_density.shape_x-2, dm_density.shape_y-2))
    ti_density_map = ti.static(ti_density_grad_px)

    for i in range(start_index, end_index):
        np_strainRate_data = dm_strainRate.read_single_frame_data(i)[1:-1,1:-1]
        if to == 'vorticity':
            np_vorticity_data = np_strainRate_data[...,1,0] - np_strainRate_data[...,0,1]
            ti_vorticity.from_numpy(np_vorticity_data)

            if use_density_mask:
                np_density_data = dm_density.read_single_frame_data(i)
                ti_density_data.from_numpy(np_density_data)
                dm_density.Sobel_conv_ti(1, ti_density_data, ti_density_grad_px, ts.Grid_Data.PARTIAL_X)
                dm_density.Sobel_conv_ti(1, ti_density_data, ti_density_grad_py, ts.Grid_Data.PARTIAL_Y)
                ts.ker_entry_wise_grad_mag(ti_density_grad_px, ti_density_grad_py, ti_density_grad_map)

                ts.ker_normalize(ti_density_grad_map)
                ts.ker_invBinary(ti_density_grad_map, 0.2)        
                ts.ker_entry_wise_productEqual(ti_density_grad_map, ti_vorticity)

                ti_density_map.from_numpy(np_density_data[1:-1,1:-1])
                ts.ker_normalize(ti_density_map)
                ts.ker_binary(ti_density_map, 0.5)
                ts.ker_entry_wise_productEqual(ti_density_map, ti_vorticity)

            # data.append(ti_vorticity.to_numpy())
            np.save(os.path.join(output_path, f'{attr_name}2{to}_{i}.npy'), ti_vorticity.to_numpy())
    
    # for i in range(start_index, end_index):
    #     np.save(os.path.join(output_path, f'{attr_name}2{to}_{i}.npy'), data[i-start_index])
        


def process_vel_to_strainRate(input_path, output_path, start_index, end_index, cell_size, use_density_mask=False, further_to='vorticity'):
    data = []
    dm_vel = ts.Grid_Data(input_path, 'velocity', start_index, end_index)
    dm_density = ts.Grid_Data(input_path, 'density', start_index, end_index)

    ti_density_data = ti.field(dtype=ti.f32, shape=(dm_density.shape_x, dm_density.shape_y))
    ti_density_grad_px = ti.field(dtype=ti.f32, shape=(dm_density.shape_x-2, dm_density.shape_y-2))
    ti_density_grad_py = ti.field(dtype=ti.f32, shape=(dm_density.shape_x-2, dm_density.shape_y-2))
    ti_density_grad_map = ti.field(dtype=ti.f32, shape=(dm_density.shape_x-2, dm_density.shape_y-2))
    ti_density_map = ti.static(ti_density_grad_px)
    
    ti_conved_val    = ti.field(dtype=ti.f32, shape=(dm_vel.shape_x,   dm_vel.shape_y))
    ti_conv_val_pupx = ti.field(dtype=ti.f32, shape=(dm_vel.shape_x-2, dm_vel.shape_y-2))
    ti_conv_val_pupy = ti.field(dtype=ti.f32, shape=(dm_vel.shape_x-2, dm_vel.shape_y-2))
    ti_conv_val_pvpx = ti.field(dtype=ti.f32, shape=(dm_vel.shape_x-2, dm_vel.shape_y-2))
    ti_conv_val_pvpy = ti.field(dtype=ti.f32, shape=(dm_vel.shape_x-2, dm_vel.shape_y-2))

    if further_to == 'vorticity':
        x1, y1 = 1,0
        x2, y2 = 0,1

    for i in range(start_index, end_index):
        np_vel_data = dm_vel.read_single_frame_data(i)

        np_conv_val = np.zeros((dm_vel.shape_x-2, dm_vel.shape_y-2,2,2))
        
        ti_conved_val.from_numpy(np_vel_data[...,ts.Grid_Data.VAL_X])
        dm_vel.Sobel_conv_ti(cell_size, ti_conved_val, ti_conv_val_pupx, ts.Grid_Data.PARTIAL_X)
        dm_vel.Sobel_conv_ti(cell_size, ti_conved_val, ti_conv_val_pupy, ts.Grid_Data.PARTIAL_Y)
        ti_conved_val.from_numpy(np_vel_data[...,ts.Grid_Data.VAL_Y])
        dm_vel.Sobel_conv_ti(cell_size, ti_conved_val, ti_conv_val_pvpx, ts.Grid_Data.PARTIAL_X)
        dm_vel.Sobel_conv_ti(cell_size, ti_conved_val, ti_conv_val_pvpy, ts.Grid_Data.PARTIAL_Y)

        if use_density_mask:
            np_density_data = dm_density.read_single_frame_data(i)
            ti_density_data.from_numpy(np_density_data)
            dm_density.Sobel_conv_ti(cell_size, ti_density_data, ti_density_grad_px, ts.Grid_Data.PARTIAL_X)
            dm_density.Sobel_conv_ti(cell_size, ti_density_data, ti_density_grad_py, ts.Grid_Data.PARTIAL_Y)
            ts.ker_entry_wise_grad_mag(ti_density_grad_px, ti_density_grad_py, ti_density_grad_map)

            ts.ker_normalize(ti_density_grad_map)
            ts.ker_invBinary(ti_density_grad_map, 0.2)        
            # TODO output density mask
            ts.ker_entry_wise_productEqual(ti_density_grad_map, ti_conv_val_pupx)
            ts.ker_entry_wise_productEqual(ti_density_grad_map, ti_conv_val_pupy)
            ts.ker_entry_wise_productEqual(ti_density_grad_map, ti_conv_val_pvpx)
            ts.ker_entry_wise_productEqual(ti_density_grad_map, ti_conv_val_pvpy)

            ti_density_map.from_numpy(np_density_data[1:-1,1:-1])
            ts.ker_normalize(ti_density_map)
            ts.ker_binary(ti_density_map, 0.5)
            ts.ker_entry_wise_productEqual(ti_density_map, ti_conv_val_pupx)
            ts.ker_entry_wise_productEqual(ti_density_map, ti_conv_val_pupy)
            ts.ker_entry_wise_productEqual(ti_density_map, ti_conv_val_pvpx)
            ts.ker_entry_wise_productEqual(ti_density_map, ti_conv_val_pvpy)

        np_conv_val[...,ts.Grid_Data.VAL_X, ts.Grid_Data.PARTIAL_X] = ti_conv_val_pupx.to_numpy()
        np_conv_val[...,ts.Grid_Data.VAL_X, ts.Grid_Data.PARTIAL_Y] = ti_conv_val_pupy.to_numpy()
        np_conv_val[...,ts.Grid_Data.VAL_Y, ts.Grid_Data.PARTIAL_X] = ti_conv_val_pvpx.to_numpy()
        np_conv_val[...,ts.Grid_Data.VAL_Y, ts.Grid_Data.PARTIAL_Y] = ti_conv_val_pvpy.to_numpy()

        # data.append(np_conv_val)
        np.save(os.path.join(output_path, f'vel2strainRate_{i}.npy'), np_conv_val)

        if further_to == 'vorticity':
            measured_value = (np_conv_val[:,:,x1,y1] - np_conv_val[:,:,x2,y2])
            np.save(os.path.join(output_path, f'vel2{further_to}_{i}.npy'), measured_value)

    # for i in range(start_index, end_index):
    #     np.save(os.path.join(output_path, f'vel2strainRate_{i}.npy'), data[i-start_index])
    
    # if further_to == 'vorticity':
    #     x1, y1 = 1,0
    #     x2, y2 = 0,1
    #     measured_value = (np.array(data)[:,:,:,x1,y1] - np.array(data)[:,:,:,x2,y2])
    #     for i in range(start_index, end_index):
    #         np.save(os.path.join(output_path, f'vel2{further_to}_{i}.npy'), measured_value[i-start_index])


def process_minus(input_path, output_path, start_index, end_index, attr_name_1, attr_name_2):
    data_1 = []
    data_2 = []
    for i in range(start_index, end_index):
        data_1.append(np.load(os.path.join(input_path, f'{attr_name_1}_{i}.npy')))
        data_2.append(np.load(os.path.join(input_path, f'{attr_name_2}_{i}.npy')))
    np_data_1 = np.array(data_1)
    np_data_2 = np.array(data_2)
    np_data = np_data_1 - np_data_2
    for i in range(start_index, end_index):
        np.save(os.path.join(output_path, f'{attr_name_1}MINUS{attr_name_2}_{i}.npy'), np_data[i-start_index])



def process_hist(input_path, output_path, start_index, end_index, attr_name, range_min, range_max, bins=128):
    """
    Processes histograms and saves the counts of samples in each bin as numpy arrays.

    Parameters:
    input_path (str): Path to the input data files.
    output_path (str): Path where the histogram numpy arrays will be saved.
    start_index (int): Starting index for file processing.
    end_index (int): Ending index for file processing.
    attr_name (str): Attribute name used in the file naming convention.
    range_min (float): Minimum range for histogram.
    range_max (float): Maximum range for histogram.
    bins (int, optional): Number of bins for the histogram. Defaults to 128.
    """
    for i in range(start_index, end_index):
        file_path = os.path.join(input_path, f'{attr_name}_{i}.npy')
        if os.path.exists(file_path):
            data = np.load(file_path)
            hist, _ = np.histogram(data, bins=bins, range=(range_min, range_max))
            
            output_file_path = os.path.join(output_path, f'hist_{attr_name}_{i}.npy')
            np.save(output_file_path, hist)
        else:
            print(f"File not found: {file_path}")


def process_abs(input_path, output_path, start_index, end_index, attr_name):
    """
    Processes numpy files by taking the absolute value of each element
    and saves the result to a new file.

    Parameters:
    input_path (str): Path to the input data files.
    output_path (str): Path where the processed files will be saved.
    start_index (int): Starting index for file processing.
    end_index (int): Ending index for file processing.
    attr_name (str): Attribute name used in the file naming convention.
    """

    for i in range(start_index, end_index):
        file_path = os.path.join(input_path, f'{attr_name}_{i}.npy')
        if os.path.exists(file_path):
            data = np.load(file_path)
            abs_data = np.abs(data)
            
            output_file_path = os.path.join(output_path, f'abs_{attr_name}_{i}.npy')
            np.save(output_file_path, abs_data)
        else:
            print(f"File not found: {file_path}")

