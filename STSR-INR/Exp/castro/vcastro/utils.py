import os
import numpy as np
import yaml
import torch
from pathlib import Path
import shutil
import time
from tqdm import tqdm
from logger import *
import torch.nn.functional as F
import sys
import json
import pickle
import yt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def flattenDict(d):
    """flatten the nesting dict to a flatten dict
    """
    flatten_d = {}
    for key, value in d.items():
        if isinstance(value, dict):
            flatten_d.update(flattenDict(value))
        else:
            flatten_d[key] = value
    return flatten_d

def interpolation_embedding(embedding_weight,key_indexs):
    #* embedding weight should be torch tensor object
    for i in range(1,len(key_indexs)):
        index_e = key_indexs[i]
        index_s = key_indexs[i-1]
        for index_i in range(index_s+1,index_e):
            d_si = index_i - index_s
            d_ie = index_e - index_i
            d_se = index_e - index_s
            embedding_weight[index_i-1] = (d_ie/d_se) * embedding_weight[index_s-1] + (d_si/d_se) * embedding_weight[index_e-1]
    total_time_steps = embedding_weight.shape[0]
    
    if key_indexs[-1] != total_time_steps:
        index_e = key_indexs[-1]
        index_s = key_indexs[-2]
        for index_i in range(key_indexs[-1]+1,total_time_steps+1):
            d_si = index_i - index_s
            d_ie = index_e - index_i
            d_se = index_e - index_s
            embedding_weight[index_i-1] = (d_ie/d_se) * embedding_weight[index_s-1] + (d_si/d_se) * embedding_weight[index_e-1]
    return embedding_weight

def getYMD():
    """return current time with format YearMonthDay (e.g. 20220928)

    Returns:
        YMDString(str): the format string YearMonthDay
    """
    timeArray = time.localtime(time.time())
    YMDString = time.strftime(r"%Y%m%d",timeArray)
    return YMDString

def json_loader(file_path):
    content = None
    with open(file_path,'r') as f:
        content = json.load(f)
    return content


def yaml_loader(file_path):
    settings = yaml.safe_load(open(file_path))
    return settings

def ensure_dirs(dir_path,verbose = False):
    upperDir = os.path.dirname(dir_path)
    if not os.path.exists(dir_path):
        if not os.path.exists(upperDir):
            ensure_dirs(upperDir,verbose=verbose)
        if verbose: print(f'{dir_path} not exists, create the dir')
        os.mkdir(dir_path)
    else:
        if verbose: print(f'{dir_path} exists, no need to create the dir')

def delFilesInDir(dir_path):
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path,f))

def TypeA2TypeB(dir_path,saved_dir_path,typeA='.dat',typeB='.raw'):
    """this function is outdated version write before 12/09/2022"""
    ensure_dirs(saved_dir_path) #create the saved dir path if the path do not exists
    delFilesInDir(saved_dir_path)
   
    for root,dirs,files in os.walk(dir_path):
        for file_name in files:
            if(os.path.splitext(file_name)[-1] == typeA): #to avoid other types of files
                dat_file_path = os.path.join(root,file_name)
                raw_file_path = os.path.join(saved_dir_path,file_name)
                shutil.copyfile(dat_file_path,raw_file_path) #first copy the dat file to RAW dir

    for root,dirs,files in os.walk(saved_dir_path):
        for file_name in files:
            if(os.path.splitext(file_name)[-1] == typeA): #just to avoid conner case
                dat_file = os.path.join(root,file_name)
                raw_file = os.path.join(root,os.path.splitext(file_name)[0]+typeB)
                os.rename(dat_file,raw_file)


# def readDat(file_path):
#     "basic & core func"
#     dat = np.fromfile(file_path,dtype='<f')
#     return dat

def readDat(file_path):
    """basic & core func"""
    if "nyx" in file_path or "castro" in file_path:  # Add specific logic for nyx if needed
        # For example, nyx might need a different dtype or processing step
        dat = np.fromfile(file_path, dtype='<f4')
    else:
        dat = np.fromfile(file_path, dtype='<f')
    return dat

def getPSNRFile(GT_dat_file,eval_dat_file):
    dat_GT = readDat(GT_dat_file)
    dat_eval = readDat(eval_dat_file)
    GT_range = np.max(dat_GT) - np.min(dat_GT)
    MSE = np.mean((dat_eval- dat_GT)**2)
    PSNR = 20*np.log10(GT_range) - 10*np.log10(MSE)
    return PSNR



def getAllDatFilePaths(dir_path, extension='.raw'):
    """Get all file paths with a specific extension in dir_path.

    Args:
        dir_path (str): The directory path to search.
        extension (str): The file extension to look for (e.g., '.raw', '.pkl').

    Returns:
        filePaths (list): A list containing all file paths with the specified extension in the dir_path.
    """
    filePaths = []
    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            if file_name.endswith(extension):
                filepath = os.path.join(root, file_name)
                filePaths.append(filepath)
    filePaths.sort()
    return filePaths

def copy_modelSetting(log_base_dir,copy_file_names=None):
    """save model.py, main.py to log_base_dir

    Args:
        log_base_dir (_type_): _description_
        copy_file_names (list(str)): a list of file names want to save in log file (main.py etc)
    """
    copy_file_path = []
    if copy_file_names == None:
        dirs_and_files = os.listdir('.')
        for item in dirs_and_files:
            rel_path = os.path.join('.',item)
            if os.path.isfile(rel_path):
                copy_file_path.append(rel_path)
    else:
        for item in copy_file_names:
            copy_file_path.append(os.path.join('.',item))

    for one_path in copy_file_path:
        dst_path = os.path.join(log_base_dir,one_path)
        if os.path.exists(dst_path):
            os.remove(dst_path)
        shutil.copyfile(one_path,dst_path)


def getLatestModelPath(model_dir,verbose=False):
    """Get the latest model path from model_dir based on edit time
    (Attention: It would be better that you explicitly input the latest model path. But to make sure 
    this method work correctly, you need to turn on verbose to check the whether the file path is correct)

    Args:
        model_dir (str): the dir path which contains all the model files
    Returns:
        model_path(str): the latest model file path
    """
    latest_file_path = None
    latest_time = 0
    last_file_name = None

    # model_dir = "./Exp/nyx/vnyx/Log"
    model_dir = "./Exp/castro/vcastro/Log"
    print("model_dir", model_dir)
    # last_file_name = os.path.join(model_dir, "STSR-nyx_Scalar-600.pth")
    # last_file_name = os.path.join(model_dir, "STSR-nyx_Scalar-100.pth")
    last_file_name = os.path.join(model_dir, "STSR-castro_Scalar-100.pth")
    print(last_file_name)

    latest_file_path = last_file_name

    # for root,dirs,files in os.walk(model_dir):
    #     for file_name in files:
    #         _,ext = os.path.splitext(file_name)
    #         if ext != '.pth':
    #             continue
    #         file_path = os.path.join(root,file_name)
    #         modifyTime = os.path.getmtime(file_path)
    #         if(modifyTime > latest_time):
    #             last_file_name = file_name
    #             latest_time = modifyTime
    #             latest_file_path = file_path
    # if last_file_name == None:
    #     raise ValueError(f"getLatestModelPath: Result is {last_file_name}. Can not get the latest model, check your model_dir argument")
    # if verbose:
    #     print(f'getLatestModelPath: load latest file path {last_file_name}')

    return latest_file_path


class EvalMetric():
    def __init__(self,GT_dirPath,eval_dirPath,data_type='3D',verbose=False):
        self.GT_dirPath = GT_dirPath
        self.eval_dirPath = eval_dirPath
        self.verbose = verbose
        self.data_type = data_type
        self.PSNR = []
        self.SSIM = []
        
    def getPSNRFile(self,GT_dat_file,eval_dat_file):
        """return the PSNR of one file

        Args:
            GT_dat_file (str): The GT file path
            eval_dat_file (str): the eval data file path

        Returns:
            _type_: _description_
        """
        dat_GT = readDat(GT_dat_file)
        dat_eval = readDat(eval_dat_file)
        GT_range = np.max(dat_GT) - np.min(dat_GT)
        MSE = np.mean((dat_eval- dat_GT)**2)
        PSNR = 20*np.log10(GT_range) - 10*np.log10(MSE)
        return PSNR
    

    def getPSNR(self,GT_length=None):
        """get PSNR from the GT and eval data
        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # self.GT_dirPath = "../Datasets/Nyx/nyx_ensemble_test/Nyx/Exec/LyA/STSRINR_GT"
        self.GT_dirPath = "../Datasets/Castro/Exec/science/wdmerger/STSRINR_GT"
        print(self.GT_dirPath, self.eval_dirPath)
        # nyx_ensemble_test/Nyx/Exec/LyA/STSRINR_GT
        filePaths_GT = getAllDatFilePaths(self.GT_dirPath)
        filePaths_eval = getAllDatFilePaths(self.eval_dirPath)
        print("filePaths_GT", filePaths_GT)
        print("filePaths_eval", filePaths_eval)
        GT_length = len(filePaths_eval)
        for index in tqdm(range(GT_length),desc="Calculating Metrics",disable=(not self.verbose)):
            self.PSNR.append(self.getPSNRFile(filePaths_GT[index],filePaths_eval[index]))
        
        MeanPSNR = np.array(self.PSNR).mean()


        return MeanPSNR, self.PSNR

class MetricsCalculator:
    def __init__(self, GT_filePath, eval_dirPath, verbose=False):
        self.GT_filePath = GT_filePath
        self.eval_dirPath = eval_dirPath
        self.verbose = verbose
        self.PSNR = []

    def load_GT_data(self):
        """Load ground truth data from the GT file."""
        with open(self.GT_filePath, 'rb') as f:
            GT_data = pickle.load(f)
        GT_data = GT_data[:, 0]  # Use only the density field
        GT_data = np.asfortranarray(GT_data)
        return GT_data

    def load_eval_data(self, eval_file):
        """Load evaluation data from a .raw file."""
        with open(eval_file, 'rb') as f:
            eval_data = np.fromfile(f, dtype=np.float32)
        eval_data = eval_data.reshape(-1, 128, 128, 128, order='F')
        return eval_data

    def normalize_data(self, data):
        """Normalize data to the range [0, 1]."""
        min_val = np.min(data)
        max_val = np.max(data)
        norm_data = (data - min_val) / (max_val - min_val)
        return norm_data

    def calculate_psnr(self, gt, pred):
        """Calculate PSNR between two numpy arrays."""
        mse = np.mean((gt - pred) ** 2)
        print(mse)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0  # Assuming the pixel values are normalized to [0, 1]
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
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

    def getPSNR(self):
        """Get PSNR from the GT and eval data."""
        # Load GT data
        GT_data = self.load_GT_data()
        print(f"GT Data Loaded: shape={GT_data.shape}, min={GT_data.min()}, max={GT_data.max()}, mean={GT_data.mean()}")

        # Fetch eval file paths
        filePaths_eval = getAllDatFilePaths(self.eval_dirPath, extension='.raw')
        print("filePaths_eval", len(filePaths_eval))

        # Check that GT_data can be divided into segments corresponding to eval files
        if len(GT_data) % len(filePaths_eval) != 0:
            raise ValueError("GT data length is not divisible by the number of eval files")

        segment_length = len(GT_data) // len(filePaths_eval)

        # Calculate PSNR for each file pair
        for index in tqdm(range(len(filePaths_eval)), desc="Calculating Metrics", disable=(not self.verbose)):
            eval_data = self.load_eval_data(filePaths_eval[index])
            print(f"Eval Data Loaded (Index {index}): shape={eval_data.shape}, min={eval_data.min()}, max={eval_data.max()}, mean={eval_data.mean()}")

            gt_segment = GT_data[index * segment_length : (index + 1) * segment_length]
            print(f"GT Segment (Index {index}): shape={gt_segment.shape}, min={gt_segment.min()}, max={gt_segment.max()}, mean={gt_segment.mean()}")
            
            # Normalize the data before calculating PSNR
            gt_segment_norm = self.normalize_data(gt_segment[0])
            eval_data_norm = self.normalize_data(eval_data[0])
            
            print(f"Normalized GT Segment (Index {index}): min={gt_segment_norm.min()}, max={gt_segment_norm.max()}, mean={gt_segment_norm.mean()}")
            print(f"Normalized Eval Data (Index {index}): min={eval_data_norm.min()}, max={eval_data_norm.max()}, mean={eval_data_norm.mean()}")

            # Render the volumes
            self.render_volume_yt(gt_segment[0], 'renders', f'gt_segment_{index}.png')
            self.render_volume_yt(eval_data[0], 'renders', f'eval_data_{index}.png')
            
            psnr_value = self.calculate_psnr(gt_segment_norm, eval_data_norm)  # Use only the density data
            self.PSNR.append(psnr_value)

        MeanPSNR = np.array(self.PSNR).mean()
        return MeanPSNR, self.PSNR
    
    
def getAllDatFilePaths(dir_path, extension='.raw'):
    """Get all file paths with a specific extension in dir_path.

    Args:
        dir_path (str): The directory path to search.
        extension (str): The file extension to look for (e.g., '.raw', '.pkl').

    Returns:
        filePaths (list): A list containing all file paths with the specified extension in the dir_path.
    """
    filePaths = []
    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            if file_name.endswith(extension):
                filepath = os.path.join(root, file_name)
                filePaths.append(filepath)
    filePaths.sort()
    return filePaths

def getMemorySize(verbose=True,**kwargs):
	"""get the memory cost of variables,
	use like this: getMemorySize(a=sth,b=sth) (sth means the the variable you want to check its mem cost)

	Args:
		verbose (bool, optional): _description_. Defaults to True.

	Raises:
		TypeError: _description_

	Returns:
		_type_: _description_
	"""
	var_names_ls = []
	var_instances_ls = []
	for k,v in kwargs.items():
		var_names_ls.append(k)
		var_instances_ls.append(v)

	mem_cost_ls = []

	for i in range(len(var_names_ls)):
		var_name = var_names_ls[i]
		var = var_instances_ls[i]
		if not isinstance(var,type(torch.tensor([1]))):
			raise TypeError(f'{type(var)} not support')
		mem_cost = sys.getsizeof(var.storage()) #unit Byte
		mem_cost /= 1024*1024 #unit MB
		mem_cost_ls.append(mem_cost)
		if verbose:
			print(f"the {var_name} take a memory cost of {mem_cost:.2f} MB")
	
	return mem_cost_ls

def get_mgrid(sidelen, dim=2, s=1,t=0):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[1]:s, :sidelen[0]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[1] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[0] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[2]:s, :sidelen[1]:s, :sidelen[0]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[2] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[0] - 1)
    elif dim == 4:
        pixel_coords = np.stack(np.mgrid[:sidelen[0]:(t+1), :sidelen[3]:s, :sidelen[2]:s, :sidelen[1]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[3] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
        pixel_coords[..., 3] = pixel_coords[..., 3] / (sidelen[1] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)
    pixel_coords -= 0.5
    pixel_coords *= 2.
    # print(pixel_coords.shape)
    pixel_coords = np.reshape(pixel_coords,(-1,dim))
    return pixel_coords

# def parseDatasetInfo(dataset_varList,dataHeaderJsonPath):
#     #*dataset should have a form like DatasetName_VarName
#     def getVarDatasetInfo(DatasetName,VarName,dataHeaderJsonPath):
#         dataHeader = json_loader(dataHeaderJsonPath)
#         varInfo = dataHeader[DatasetName]
#         del varInfo['vars']
#         varInfo['data_path'] = varInfo['data_path'][VarName]
#         # varInfo['norm_data_path'] = varInfo['norm_data_path'][VarName]
#         return varInfo
#     d = {dataset_var:{} for dataset_var in dataset_varList}
#     for i in dataset_varList:
#         datasetName,varName = i.split("_")
#         d[i]=getVarDatasetInfo(datasetName,varName,dataHeaderJsonPath)
#     return d

def parseDatasetInfo(dataset_varList, dataHeaderJsonPath):
    def getVarDatasetInfo(DatasetName, VarName, dataHeaderJsonPath):
        dataHeader = json_loader(dataHeaderJsonPath)
        varInfo = dataHeader.get(DatasetName, {})
        if 'vars' in varInfo:
            del varInfo['vars']
        if DatasetName == 'nyx' or DatasetName == 'castro':  # Add any specific logic for 'nyx' dataset here
            print(varInfo)
            varInfo['data_path'] = varInfo['data_path'].get(VarName, None)
            # input("x")
        else:
            varInfo['data_path'] = varInfo['data_path'].get(VarName, None)
        return varInfo

    d = {dataset_var: {} for dataset_var in dataset_varList}
    for i in dataset_varList:
        print("i", i)
        datasetName, varName = i.split("_")
        d[i] = getVarDatasetInfo(datasetName, varName, dataHeaderJsonPath)
    return d

if __name__ == "__main__":
    test_ymlPath = ".\\tasks\\configs\\multivars.yml"
    a=yaml_loader(test_ymlPath)['data_setting']['dataset']
    print(parseDatasetInfo(a,".\\dataInfo\\localDataInfo.json"))
    

    
    
   