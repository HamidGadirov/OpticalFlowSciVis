import os
import numpy as np
from typing import List
from multiprocessing import Process
import taichi as ti

class SeqData:
    def __init__(self, path, attr_name, compressed=False):
        self.path = path
        self.attr_name = attr_name
        self.compressed = compressed
        self.len, self.start_index, self.end_index = self.inspect()
    
    def inspect(self):
        files = os.listdir(self.path)
        if self.compressed:
            relevant_files = [f for f in files if f.startswith(self.attr_name) and f.endswith(".npz")]
        else:
            relevant_files = [f for f in files if f.startswith(self.attr_name) and f.endswith(".npy")]
        indices = [int(f.split('_')[-1].split('.')[0]) for f in relevant_files]
        indices.sort()
        start_number = min(indices) if indices else None
        end_number = max(indices) if indices else None

        # Check for missing files
        missing_files = []
        for i in range(max(indices) + 1):
            if i not in indices:
                missing_files.append(f"{self.attr_name}_{i}.npy(z)")
        print(f"SeqData.inspect(): In {self.path}, found {len(relevant_files)} files for attribute {self.attr_name}, ranging [{start_number}, {end_number}].")
        if len(missing_files) > 0:
            raise Exception(f"Missing files for attribute {self.attr_name}: {missing_files}")
        
        return len(relevant_files), start_number, end_number
    
    def reshape_to_3d(self, index_attr_name:str, output_path:str, output_name:str, order=[0,1,2], compressed=False, offset=0):
        if self.compressed:
            index_template_zip = np.load(os.path.join(self.path, index_attr_name+'.npz'))
            index_template = index_template_zip[list(index_template_zip.keys())[0]]
        else:
            index_template = np.load(os.path.join(self.path, index_attr_name+'.npy'))
        for i in range(self.start_index, self.end_index+1):
            if self.compressed:
                raw_data_zip = np.load(os.path.join(self.path, self.attr_name+'_'+str(i)+'.npz'))
                raw_data = raw_data_zip[list(raw_data_zip.keys())[0]]
            else:
                raw_data = np.load(os.path.join(self.path, self.attr_name+'_'+str(i)+'.npy'))
            reshaped_data = None
            xNodeNum = index_template[:,order[0]].max() + 1
            yNodeNum = index_template[:,order[1]].max() + 1
            zNodeNum = index_template[:,order[2]].max() + 1
            raw_data_shape_dim = len(raw_data.shape)
            
            if raw_data_shape_dim == 1:
                reshaped_data = np.empty((xNodeNum, yNodeNum, zNodeNum), dtype=float)
                reshaped_data[index_template[:,order[0]], index_template[:,order[1]], index_template[:,order[2]]] = raw_data
            elif raw_data_shape_dim == 2:
                reshaped_data = np.empty((xNodeNum, yNodeNum, zNodeNum, raw_data.shape[1]), dtype=float)
                for j in range(raw_data.shape[1]):
                    reshaped_data[index_template[:,order[0]], index_template[:,order[1]], index_template[:,order[2]], j] = raw_data[:,j]
            elif raw_data_shape_dim == 3:
                reshaped_data = np.empty((xNodeNum, yNodeNum, zNodeNum, raw_data.shape[1], raw_data.shape[2]), dtype=float)
                for j in range(raw_data.shape[1]):
                    for k in range(raw_data.shape[2]):
                        reshaped_data[index_template[:,order[0]], index_template[:,order[1]], index_template[:,order[2]], j, k] = raw_data[:,j,k]
            elif raw_data_shape_dim > 3:
                raise Exception("raw_data.shape is not supported")
            if compressed:
                np.savez_compressed(os.path.join(output_path, output_name+'_'+str(i+offset-self.start_index)+'.npz'), reshaped_data)
            else:
                np.save(os.path.join(output_path, output_name+'_'+str(i+offset-self.start_index)+'.npy'), reshaped_data)
            

class Grid_Data_manager:
    def __init__(self, input_folder_path: str, output_folder_path: str) -> None:
        self.input_folder_path = os.path.abspath(input_folder_path)
        self.output_folder_path = os.path.abspath(output_folder_path)
        self.start_index = None
        self.end_index = None
        self.attr_name = None
        self.raw_data = []
        self.processed_data = []
    
    def read_data(self, attr:str, start_index:int, end_index: int):
        self.attr_name = attr
        self.start_index = start_index
        self.end_index = end_index
        for i in range(start_index, end_index):
            file_name = attr+'_'+str(i)+'.npy'
            file_path = os.path.join(self.input_folder_path, file_name)
            data_arr = np.load(file_path)
            self.raw_data.append(data_arr)
    
    def export_data(self, name:str='data'):
        exported_data = self.processed_data
        np.save(os.path.join(self.output_folder_path, name+'.npy'), exported_data)
        return exported_data
    
    def export_single_frame_data(self, name:str='data', operations:List[str]=[]):
        length = len(operations)+1
        for i in range(self.start_index, self.end_index):
            exported_data_list = []
            exported_data = self.processed_data[i-self.start_index]
            exported_data_list.append(exported_data)

            for operation in operations:
                if operation == 'flipud':
                    exported_data_list.append(np.flipud(exported_data))
                elif operation == 'fliplr':
                    exported_data_list.append(np.fliplr(exported_data))
                elif operation == 'transpose':
                    if len(exported_data.shape)==2:
                        exported_data_list.append(np.transpose(exported_data))
                    elif len(exported_data.shape)==3:
                        exported_data_list.append(np.transpose(exported_data, (1,0,2)))
                    elif len(exported_data.shape)==4:
                        exported_data_list.append(np.transpose(exported_data, (1,0,2,3)))
                    else:
                        raise Exception("exported_data.shape is not supported")
                elif operation == 'flipud_fliplr':
                    exported_data_list.append(np.flipud(np.fliplr(exported_data)))
            
            for j in range(length):
                np.save(os.path.join(self.output_folder_path, name+'_'+str((i-self.start_index)*length+j)+'.npy'), exported_data_list[j])
    
    def reshape_data_to_2d(self, index_attr:str):
        for i in range(self.start_index, self.end_index):
            index_arr = self.get_index_arr_2d(index_attr, i)
            self.processed_data.append(self.reshape_data_with_index_2d(self.raw_data[i-self.start_index], index_arr))
    
    def get_index_arr_2d(self, index_attr:str, i:int):
        index_name = index_attr+'_'+str(i)+'.npy'
        index_path = os.path.join(self.input_folder_path, index_name)
        index_arr = np.load(index_path)
        return index_arr
    
    def reshape_data_with_index_2d(self, data_arr, index_arr):
        rows = index_arr[:,0].max() + 1
        cols = index_arr[:,1].max() + 1
        if len(data_arr.shape)==1:
            reshaped_data = np.empty((rows, cols), dtype=float)
            reshaped_data[index_arr[:, 0], index_arr[:, 1]] = data_arr
        elif len(data_arr.shape)==2:
            reshaped_data = np.empty((rows, cols, data_arr.shape[1]), dtype=float)
            for i in range(data_arr.shape[1]):
                reshaped_data[index_arr[:, 0], index_arr[:, 1], i] = data_arr[:, i]
        elif len(data_arr.shape)==3:
            reshaped_data = np.empty((rows, cols, data_arr.shape[1], data_arr.shape[2]), dtype=float)
            for i in range(data_arr.shape[1]):
                for j in range(data_arr.shape[2]):
                    reshaped_data[index_arr[:, 0], index_arr[:, 1], i, j] = data_arr[:, i, j]
        else:
            raise Exception("data_arr.shape is not supported")

        return reshaped_data

@ti.data_oriented
class Grid_Data:
    FRAME = 0
    SHAPE_X = 0
    SHAPE_Y = 1
    VAL_X = 0
    VAL_Y = 1
    PARTIAL_X = 0
    PARTIAL_Y = 1

    # changing to the Cartesian coordinate system
    Soble_X = np.transpose(np.flip(np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]]), axis=0))
    
    Soble_Y = np.transpose(np.flip(np.array([[ 1,  2,  1],
                                [ 0,  0,  0],
                                [-1, -2, -1]]), axis=0))
    
    TI_Soble_X = ti.Matrix([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
    
    TI_Soble_Y = ti.Matrix([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]])


    def __init__(self, path, attr_name, start_index: int, end_index: int) -> None:
        self.path = path
        self.attr_name = attr_name
        self.start_index = start_index
        self.end_index = end_index
        self.shape_x = None
        self.shape_y = None
        
        if not os.path.exists(path):
            raise Exception("path does not exist")
        if not os.path.isdir(path):
            raise Exception("path is not a folder")
        if start_index < 0 or end_index < 0:
            raise Exception("start_index or end_index is negative")
        if start_index > end_index:
            raise Exception("start_index is larger than end_index")

        self.shape_x = self.read_single_frame_data(start_index).shape[self.SHAPE_X]
        self.shape_y = self.read_single_frame_data(start_index).shape[self.SHAPE_Y]
    
    def read_single_frame_data(self, frame):
        if frame < self.start_index or frame > self.end_index:
            raise Exception("frame is out of range")
        file_name = self.attr_name+'_'+str(frame)+'.npy'
        file_path = os.path.join(self.path, file_name)
        if not os.path.exists(file_path):
            raise Exception("file does not exist")
        np_single_frame_data = np.load(file_path)
        return np_single_frame_data
    
    def read_all_frame_data(self):
        np_all_frame_data = []
        for i in range(self.start_index, self.end_index):
            np_all_frame_data.append(self.read_single_frame_data(i))
        return np_all_frame_data

    def get_3x3_data(self, np_single_frame_data, grid_x, grid_y):
        # if (not 0 < grid_x < self.shape_x-1) or (not 0 < grid_y < self.shape_y-1):
        #     raise Exception("grid_x or grid_y is out of range")
        return np_single_frame_data[grid_x-1:grid_x+2, grid_y-1:grid_y+2]
    
    @ti.func
    def get_3x3_data_ti(self, ti_single_frame_data:ti.types.ndarray(), grid_x, grid_y):
        return ti_single_frame_data[grid_x-1:grid_x+2, grid_y-1:grid_y+2]
    
    def get_single_data(self, np_single_frame_data, grid_x, grid_y):
        # if (not -1 < grid_x < self.shape_x) or (not -1 < grid_y < self.shape_y):
        #     raise Exception("grid_x or grid_y is out of range")
        return np_single_frame_data[grid_x, grid_y]
    
    def Sobel_conv(self, np_single_frame_data):
        channel_num = np_single_frame_data.shape[2]
        np_conv_val = np.zeros((self.shape_x-2, self.shape_y-2, channel_num, 2))

        for index_x in range(1, self.shape_x-1):
            for index_y in range(1, self.shape_y-1):
                for channel in range(channel_num):
                    np_conv_val[index_x-1, index_y-1, channel, 0] = \
                        np.sum(self.get_3x3_data(np_single_frame_data, index_x, index_y)[...,channel] * self.Soble_X)
                    np_conv_val[index_x-1, index_y-1, channel, 1] = \
                        np.sum(self.get_3x3_data(np_single_frame_data, index_x, index_y)[...,channel] * self.Soble_Y)
        return np_conv_val
        
    @ti.kernel
    def Sobel_conv_ti(self, cell_size:ti.f32, ti_single_frame_data:ti.template(), ti_conv_val: ti.template(), partial: ti.i32):
        for index_x in range(1, self.shape_x-1):
            for index_y in range(1, self.shape_y-1):
                data3x3=ti.Matrix([[ti_single_frame_data[index_x-1,index_y-1],ti_single_frame_data[index_x  ,index_y-1],ti_single_frame_data[index_x+1,index_y-1]],
                                   [ti_single_frame_data[index_x-1,index_y  ],ti_single_frame_data[index_x  ,index_y  ],ti_single_frame_data[index_x+1,index_y  ]],
                                   [ti_single_frame_data[index_x-1,index_y+1],ti_single_frame_data[index_x  ,index_y+1],ti_single_frame_data[index_x+1,index_y+1]]])
                if partial == self.PARTIAL_X:
                    ti_conv_val[index_x-1, index_y-1] = (data3x3 * self.TI_Soble_X).sum() / cell_size / 8
                elif partial == self.PARTIAL_Y:
                    ti_conv_val[index_x-1, index_y-1] = (data3x3 * self.TI_Soble_Y).sum() / cell_size / 8

