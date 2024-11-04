import numpy as np
import torch
import skimage 
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage import data,img_as_float
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
import random
from utils import *
import pickle

#-------The lib below is not functional----------#
from pprint import pprint

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def sliceAlongAxis(array,axis):
    dims = array.shape
    if (axis==0):
        return array[0,:,:].reshape(1,dims[1],dims[2])
    elif (axis==1):
        return array[:,0,:].reshape(dims[0],1,dims[2])
    elif (axis==2):
        return array[:,:,0].reshape(dims[0],dims[1],1)
    else:
        raise ValueError(f"sliceAlongAxis: Axis {axis} not valid, should be [0,2]")
    
def complementZero(array,complementAxis):
    complementArray = np.zeros(sliceAlongAxis(array,complementAxis).shape)
    array = np.concatenate((complementArray,array),axis=complementAxis)
    return array

import json

class MultiVarDataset:
    def __init__(self, args):
        self.data_settings = args['data_setting']
        self.mode = args['mode']
        self.batch_size = self.data_settings['batch_size']
        self.datasetInfoPath = "./dataInfo/localDataInfo.json" if self.mode in ['debug', 'local_inf'] else './dataInfo/CRCDataInfo.json'
        self.datasetInfo = parseDatasetInfo(self.data_settings['dataset'], self.datasetInfoPath)
        self.dataset_varList = self.data_settings['dataset']
        self.unsupervised = self.data_settings['unsupervised']
        self.scale, self.interval = self.parseScale_Interval(self.data_settings['scale'], self.data_settings['interval'])
        self.splitTimes = self.data_settings['splitTimes']

        self.completeDatasetInfo()

    # def ReadData(self):
    #     self.data = {dataset_var: [] for dataset_var in self.dataset_varList}
    #     for d_v in self.dataset_varList:
    #         d, v = d_v.split('_')
    #         train_time_steps = self.datasetInfo[d_v]['train_time_steps']
    #         data_path = self.datasetInfo[d_v]['data_path']
    #         dim = self.datasetInfo[d_v]['dim']
    #         self.space_sampled_indices = self.Space_Subsample(dim, d_v)
    #         for i in train_time_steps:
    #             if d in ['supercurrent', 'Tornado', 'Cylinder', 'Earthquake', 'Tangaroa', 'Jet']:
    #                 v = np.fromfile(data_path + '{:04d}'.format(i) + '.raw', dtype='<f')
    #             elif d == 'nyx':  # Add support for nyx data
    #                 v = self.read_nyx_file(data_path, i)
    #             else:
    #                 v = np.fromfile(data_path + '{:04d}'.format(i) + '.raw', dtype='<f')

    #             v = 2 * (v - np.min(v)) / (np.max(v) - np.min(v)) - 1
    #             v = v[self.space_sampled_indices]
    #             self.data[d_v].append(v)

    def ReadData(self):
        self.data = {dataset_var: [] for dataset_var in self.dataset_varList}
        for d_v in self.dataset_varList:
            d, v = d_v.split('_')
            data_path = self.datasetInfo[d_v]['data_path']
            dim = self.datasetInfo[d_v]['dim']
            self.space_sampled_indices = self.Space_Subsample(dim, d_v)
            print("dim, d_v, space_sampled_indices:", dim, d_v, self.space_sampled_indices)

            if d == 'nyx' or d == 'castro':  # Handle nyx data separately
                v_data = self.read_nyx_file(data_path)
                
                if v_data.size > 0:
                    v_data = v_data[self.space_sampled_indices]
                    v_data = 2 * (v_data - np.min(v_data)) / (np.max(v_data) - np.min(v_data)) - 1
                    self.data[d_v].append(v_data)
            else:
                train_time_steps = self.datasetInfo[d_v]['train_time_steps']
                for i in train_time_steps:
                    v = np.fromfile(data_path + '{:04d}'.format(i) + '.raw', dtype='<f')
                    v = 2 * (v - np.min(v)) / (np.max(v) - np.min(v)) - 1
                    v = v[self.space_sampled_indices]
                    self.data[d_v].append(v)

    # def read_nyx_file(self, data_path):
    #     # Full path to the .pkl file
    #     # file_path = os.path.join(data_path, file_name)
    #     file_path = data_path
    #     print("read_nyx_file", file_path)
        
    #     try:
    #         with open(file_path, 'rb') as f:
    #             data = pickle.load(f)
            
    #         print("data", data.shape)
    #         v = data[:, 0]

    #         if v is None:
    #             print(f"No data found in file {file_path}")
    #             return np.array([])  # Return an empty array if no data is found
            
    #         # Process the data as needed
    #         # Normalize the data or reshape it if necessary
    #         return np.array(v)
            
    #     except FileNotFoundError:
    #         print(f"File {file_path} not found.")
    #         return np.array([])  # Handle missing data gracefully

    #     except Exception as e:
    #         print(f"An error occurred while reading the file {file_path}: {e}")
    #         return np.array([])  # Return an empty array on error

    def render_volume_yt(self, volume, save_path, file_name):
        """Render the 3D volume using yt and save the image."""
        ds_numpy = yt.load_uniform_grid(dict(density=volume), volume.shape)
        sc = yt.create_scene(ds_numpy, field=("gas", "density"))
        source = sc[0]

        source.set_log(True)
        bounds = (0.1, 0.7)
        tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        tf.sample_colormap(np.log10(0.2), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf.sample_colormap(np.log10(0.3), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf.sample_colormap(np.log10(0.4), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf.sample_colormap(np.log10(0.5), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")

        source.tfh.tf = tf
        source.tfh.bounds = bounds

        cam = sc.camera
        cam.resolution = 256
        cam.zoom(1.0)
        im = sc.render()
        im.write_png(os.path.join(save_path, file_name), background=[0.0, 0.0, 0.0, 1.0], sigma_clip=4.0) # 2.0
        print("Rendered:", file_name)
        
    def read_nyx_file(self, data_path):
        # Full path to the .pkl file
        file_path = data_path
        print("read_nyx_file", file_path)
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # if d == 'castro':
            data = data[0] # without params
            
            print("data", data.shape)
            v = data[:, 0]  # Assuming the first variable is the density field

            if v is None:
                print(f"No data found in file {file_path}")
                return np.array([])  # Return an empty array if no data is found

            # Reshape to 3D assuming data.shape is (160, 4, 128, 128, 128) and we need the first component
            v = v.reshape(-1, 128, 128, 128)  # Reshape to (160, 128, 128, 128)

            self.render_volume_yt(v[10], 'renders', f'v_10.png')

            # Column-major order (Fortran-style order='F'): z-axis, y-axis, x-axis
            # v = np.asfortranarray(v)
            
            # Flatten the data to 1D array for subsampling
            v = v.flatten(order='F')

            # volume_viz = v.reshape((-1, 128, 128, 128))
            # volume_viz = np.asfortranarray(v).reshape((160, 128, 128, 128)) # wrong
            volume_viz = v.reshape(-1, 128, 128, 128, order='F')
            # print(volume_viz.dtype)
            self.render_volume_yt(volume_viz[10], 'renders', f'train_data.png')
            # input("train_data")
            
            self.render_volume_yt(volume_viz[10], 'renders', f'train_data.png')
            
            print(f"original: min={v.min()}, max={v.max()}, mean={v.mean()}")
            # Normalize the data to be between -1 and 1
            min_val = v.min()
            max_val = v.max()
            v = 2 * (v - min_val) / (max_val - min_val) - 1
            print(f"Normalized: min={v.min()}, max={v.max()}, mean={v.mean()}")

            return v
            
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return np.array([])  # Handle missing data gracefully

        except Exception as e:
            print(f"An error occurred while reading the file {file_path}: {e}")
            return np.array([])  # Return an empty array on error

    def parseScale_Interval(self, scale_in_setting, interval_in_setting):
        interval = {}
        scale = {}
        if isinstance(interval_in_setting, int):
            interval = {dataset_var: interval_in_setting for dataset_var in self.dataset_varList}
        if isinstance(scale_in_setting, int):
            scale = {dataset_var: scale_in_setting for dataset_var in self.dataset_varList}
        if isinstance(interval_in_setting, list):
            for idx, dataset_var in enumerate(self.dataset_varList):
                interval[dataset_var] = interval_in_setting[idx]
        if isinstance(scale_in_setting, list):
            for idx, dataset_var in enumerate(self.dataset_varList):
                scale[dataset_var] = scale_in_setting[idx]
        return scale, interval

    def completeDatasetInfo(self):
        for dataset_var in self.datasetInfo:
            total_samples = self.datasetInfo[dataset_var]['total_samples']
            self.datasetInfo[dataset_var]['train_time_steps'] = [i for i in range(1, total_samples + 1, self.interval[dataset_var] + 1)]
            self.datasetInfo[dataset_var]['total_time_steps'] = [i for i in range(1, total_samples + 1)]
            self.datasetInfo[dataset_var]['embeddingIndex'] = self.dataset_varList.index(dataset_var)

    def Space_Subsample(self, dim, d_v):
        space_sampled_indices = []
        for z in range(0, dim[2], self.scale[d_v]):
            for y in range(0, dim[1], self.scale[d_v]):
                for x in range(0, dim[0], self.scale[d_v]):
                    index = (((z) * dim[1] + y) * dim[0] + x)
                    space_sampled_indices.append(index)
        space_sampled_indices = np.asarray(space_sampled_indices)
        return space_sampled_indices

    def getMultiVarTrainLoader(self):
        training_data_input = []
        training_data_output = []
        for d_v in self.dataset_varList:
            train_time_steps = self.datasetInfo[d_v]['train_time_steps']
            total_samples = self.datasetInfo[d_v]['total_samples']
            embeddingIndex = self.datasetInfo[d_v]['embeddingIndex']
            HR_dim = self.datasetInfo[d_v]['dim']
            LR_dims = [d // self.scale[d_v] for d in HR_dim]
            for index, sampled_time in enumerate(train_time_steps):
                if index < len(self.data[d_v]):  # Ensure the index is within bounds
                    output_val = self.data[d_v][index].reshape(-1, 1)
                    split_output, reverseBuffer = self.splitVolumes(data=output_val, splitTimes=self.splitTimes, dims=LR_dims)
                    if index == 0:
                        HR_split_dims = [dim * self.scale[d_v] for dim in reverseBuffer['split_dims']]
                        reverseBuffer['HR_split_dims'] = HR_split_dims
                        space_sampled_indices = self.Space_Subsample(HR_split_dims, d_v)
                        split_coords = get_mgrid([HR_split_dims[0], HR_split_dims[1], HR_split_dims[2]], dim=3, s=1)
                        split_coords = split_coords[space_sampled_indices]
                        if self.unsupervised:
                            reverseBuffer['HR_split_dims'] = [int(d * 1.75) for d in HR_split_dims]
                        self.datasetInfo[d_v]['reverseBuffer'] = reverseBuffer

                    time_col = (2 * (sampled_time - 1) / (total_samples - 1) - 1) * np.ones((split_coords.shape[0], 1))
                    embeddingIndex_col = embeddingIndex * np.ones((split_coords.shape[0], 1))
                    input_coord = np.concatenate((embeddingIndex_col, time_col, split_coords), axis=1)

                    training_data_input += list(input_coord)
                    training_data_output += list(split_output)

        training_data_input = torch.FloatTensor(np.array(training_data_input))
        training_data_output = torch.FloatTensor(np.array(training_data_output))
        data = torch.utils.data.TensorDataset(training_data_input, training_data_output)
        train_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=True)

        return train_loader

    def getMultiVarInfLoader(self):
        test_input_coords = {dataset_var: {"t": [], "coords": None} for dataset_var in self.dataset_varList}
        for d_v in self.dataset_varList:
            reverseBuffer = self.datasetInfo[d_v]['reverseBuffer']
            HR_split_dims = reverseBuffer['HR_split_dims']
            if not self.unsupervised:
                coords = get_mgrid([HR_split_dims[0], HR_split_dims[1], HR_split_dims[2]], dim=3, s=1)
            else:
                coords = get_mgrid([HR_split_dims[0], HR_split_dims[1], HR_split_dims[2]], dim=3, s=1)
            total_samples = self.datasetInfo[d_v]['total_samples']
            total_time_steps = self.datasetInfo[d_v]['total_time_steps']
            test_input_coords[d_v]['coords'] = coords
            for t in total_time_steps:
                test_input_coords[d_v]['t'].append((2 * (t - 1) / (total_samples - 1) - 1))
        return test_input_coords


# class MultiVarDataset():
#     def __init__(self,args):
#         self.data_settings = args['data_setting']
#         self.mode = args['mode']
#         self.batch_size = self.data_settings['batch_size']
#         self.datasetInfoPath = "./dataInfo/localDataInfo.json" if self.mode in ['debug', 'local_inf'] else './dataInfo/CRCDataInfo.json'
#         self.datasetInfo = parseDatasetInfo(self.data_settings['dataset'],self.datasetInfoPath)
#         self.dataset_varList = self.data_settings['dataset']
#         self.unsupervised = self.data_settings['unsupervised']
#         self.scale,self.interval = self.parseScale_Interval(self.data_settings['scale'], self.data_settings['interval'])
#         self.splitTimes = self.data_settings['splitTimes']

#         self.completeDatasetInfo()
       
        
#     def ReadData(self):
#         self.data = {dataset_var:[] for dataset_var in self.dataset_varList}
#         for d_v in self.dataset_varList:
#             d,v = d_v.split('_')
#             train_time_steps = self.datasetInfo[d_v]['train_time_steps']
#             data_path = self.datasetInfo[d_v]['data_path']
#             dim = self.datasetInfo[d_v]['dim']
#             self.space_sampled_indices = self.Space_Subsample(dim,d_v)
#             for i in train_time_steps:
#                 if d in ['supercurrent', 'Tornado']:
#                     v = np.fromfile(data_path+'{:04d}'.format(i)+'.raw',dtype = '<f')
#                 elif d in ['Cylinder', 'Earthquake', 'Tangaroa']:
#                     v = np.fromfile(data_path+'{:04d}'.format(i)+'.raw',dtype='<f')
#                 elif d in ['Jet']:
#                     v = np.fromfile(data_path+'{:04d}'.format(i)+'.raw',dtype='<f')
#                 else:
#                     v = np.fromfile(data_path+'{:04d}'.format(i)+'.raw',dtype='<f')
                
#                 v = 2*(v-np.min(v))/(np.max(v)-np.min(v)) - 1
#                 v = v[self.space_sampled_indices]
#                 self.data[d_v].append(v)
                
#     def parseScale_Interval(self,scale_in_setting,interval_in_setting):
#         interval = {}
#         scale = {}
#         if type(interval_in_setting) == int:
#             interval = {dataset_var:interval_in_setting for dataset_var in self.dataset_varList}
#         if type(scale_in_setting) == int:
#             scale = {dataset_var:scale_in_setting for dataset_var in self.dataset_varList}
#         if type(interval_in_setting) == list:
#             for idx,dataset_var in enumerate(self.dataset_varList):
#                 interval[dataset_var] = interval_in_setting[idx]    
#         if type(scale_in_setting) == list:
#             for idx,dataset_var in enumerate(self.dataset_varList):
#                 scale[dataset_var] = scale_in_setting[idx]  
#         return scale, interval
                    
#     def completeDatasetInfo(self):
#         #* set train and inf time steps for each variable
#         for dataset_var in self.datasetInfo:
#             total_samples = self.datasetInfo[dataset_var]['total_samples']
#             self.datasetInfo[dataset_var]['train_time_steps'] = [i for i in range(1,total_samples+1,self.interval[dataset_var]+1)]
#             self.datasetInfo[dataset_var]['total_time_steps'] = [i for i in range(1,total_samples+1)]
#             self.datasetInfo[dataset_var]['embeddingIndex'] = self.dataset_varList.index(dataset_var)
#         # pprint(self.datasetInfo)
        

#     def Space_Subsample(self,dim,d_v):
#         space_sampled_indices = []
#         for z in range(0,dim[2],self.scale[d_v]):
#             for y in range(0,dim[1],self.scale[d_v]):
#                 for x in range(0,dim[0],self.scale[d_v]):
#                     index = (((z) * dim[1] + y) * dim[0] + x)
#                     space_sampled_indices.append(index)
#         space_sampled_indices = np.asarray(space_sampled_indices)
#         return space_sampled_indices

    
#     def getMultiVarTrainLoader(self):
#         #* should work even dims are not same for all dataset
#         training_data_input = []
#         training_data_output = []
#         for d_v in self.dataset_varList:
#             train_time_steps = self.datasetInfo[d_v]['train_time_steps']
#             total_samples = self.datasetInfo[d_v]['total_samples']
#             embeddingIndex = self.datasetInfo[d_v]['embeddingIndex']
#             HR_dim = self.datasetInfo[d_v]['dim']
#             LR_dims = [d//self.scale[d_v] for d in HR_dim]
#             for index,sampled_time in enumerate(train_time_steps):
#                 output_val = self.data[d_v][index].reshape(-1,1)
#                 split_output,reverseBuffer = self.splitVolumes(data=output_val,splitTimes=self.splitTimes,dims=LR_dims)
#                 if index == 0:
#                     HR_split_dims = [dim*self.scale[d_v] for dim in reverseBuffer['split_dims']]
#                     reverseBuffer['HR_split_dims'] = HR_split_dims
#                     space_sampled_indices = self.Space_Subsample(HR_split_dims,d_v)
#                     split_coords = get_mgrid([HR_split_dims[0],HR_split_dims[1],HR_split_dims[2]],dim=3,s=1)
#                     split_coords = split_coords[space_sampled_indices]
#                     if self.unsupervised:
#                         reverseBuffer['HR_split_dims'] = [int(d*1.75) for d in HR_split_dims] #* d times unsample factor
#                     self.datasetInfo[d_v]['reverseBuffer'] = reverseBuffer
                
#                 time_col = (2*(sampled_time-1)/(total_samples-1)-1)*np.ones((split_coords.shape[0],1))
#                 embeddingIndex_col = embeddingIndex*np.ones((split_coords.shape[0],1))
#                 input_coord = np.concatenate((embeddingIndex_col,time_col,split_coords),axis=1)

#                 training_data_input += list(input_coord)
#                 training_data_output += list(split_output)
        
#         training_data_input = torch.FloatTensor(np.array(training_data_input))
#         training_data_output = torch.FloatTensor(np.array(training_data_output))
#         data = torch.utils.data.TensorDataset(training_data_input,training_data_output)
#         train_loader = DataLoader(dataset=data,batch_size=self.batch_size,shuffle=True)

#         return train_loader  

#     def getMultiVarInfLoader(self):
#         test_input_coords = {dataset_var:{"t":[],"coords":None} for dataset_var in self.dataset_varList}
#         for d_v in self.dataset_varList:
#             # dim = self.datasetInfo[d_v]['dim']
#             reverseBuffer = self.datasetInfo[d_v]['reverseBuffer']
#             # print(reverseBuffer)
#             HR_split_dims = reverseBuffer['HR_split_dims']
#             if not self.unsupervised:
#                 coords = get_mgrid([HR_split_dims[0],HR_split_dims[1],HR_split_dims[2]],dim=3,s=1)
#             else:
#                 coords = get_mgrid([HR_split_dims[0],HR_split_dims[1],HR_split_dims[2]],dim=3,s=1)
#             total_samples = self.datasetInfo[d_v]['total_samples']
#             total_time_steps = self.datasetInfo[d_v]['total_time_steps']
#             test_input_coords[d_v]['coords'] = coords
#             for t in total_time_steps:
#                 test_input_coords[d_v]['t'].append((2*(t - 1)/(total_samples - 1)-1))
#         return test_input_coords 
    
    def splitVolumes(self,data,splitTimes,dims):
        data_ls = [data.reshape(dims[2],dims[1],dims[0]).transpose()]
        reverseBuffer = {"split_dims":[],"split_axis_order":[],"splitTimes":splitTimes,"complemtAxis":[-1 for i in range(splitTimes*3)]}
        split_dims = dims.copy()
        # for j in range(splitTimes*3): #* this is split accross 3 axis 
        for j in range(splitTimes): 
            res = []
            split_axis = j%3
            reverseBuffer['split_axis_order'].append(split_axis)
            complementHappen = False
            for data_idx in range(len(data_ls)):
                if(data_ls[data_idx].shape[split_axis]%2 != 0):
                    data_ls[data_idx] = complementZero(data_ls[data_idx],split_axis)
                    complementHappen = True
                
                split_arrays = np.split(data_ls[data_idx],indices_or_sections=2,axis=split_axis)
                res.append(split_arrays[0])
                res.append(split_arrays[1])
            data_ls = res
            
            if complementHappen:
                reverseBuffer['complemtAxis'][j] = split_axis
                split_dims[split_axis] = (split_dims[split_axis]+1)//2
            else:
                split_dims[split_axis] = split_dims[split_axis]//2

        data_ls = [v.flatten('F').reshape(-1,1) for v in data_ls] 
        
        reverseBuffer['split_dims'] = split_dims
        reverseBuffer['split_axis_order'].reverse()
        reverseBuffer['complemtAxis'].reverse()
        res = np.concatenate(data_ls,axis=-1)
        return res,reverseBuffer  
    

def multiSplitFlips(v_ls,axis):
    for i in range(0,len(v_ls),2):
        v_ls[i] = volumeMirrorFlip(v_ls[i], axis)
    return v_ls

def volumeMirrorFlip(v,axis):
    v = np.flip(v,axis)
    return v
        

if __name__ == "__main__":
   settings = yaml_loader('./tasks/configs/multilass.yml')
   D = MultiVarDataset(settings)
   
    
    