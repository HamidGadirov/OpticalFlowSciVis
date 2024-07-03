import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from skimage.transform import resize
import os
import numpy as np
import torch.optim as optim
import time
from model import *
from tqdm import tqdm
from utils import *
from dataio import *
from logger import *
import time 

from pprint import pprint

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reverseComplementZero(array,axis,scale):
    dims = array.shape
    if (axis==0):
        return array[1*(scale):,:,:]
    elif (axis==1):
        return array[:,1*(scale):,:]
    elif (axis==2):
        return array[:,:,1*(scale):]
    else:
        raise ValueError(f"sliceAlongAxis: Axis {axis} not valid, should be [0,2]")

class Train():
    def __init__(self,model,latentTable,args,logger):
        #init and parse the settings
        self.all_setting = args
        self.mode = args['mode']
        self.model_setting = args['model_setting']
        self.train_setting = args['train_setting']
        self.num_epochs = self.train_setting['num_epochs']
        self.checkpoints_interval = self.train_setting['checkpoints_interval']
        self.data_setting = args['data_setting']
        self.model_name = args['model_name']
        self.dataset_name = self.data_setting['dataset']
        self.log_rootDir = self.train_setting['log_root_dir']
        self.log_base_dir = args['log_base_dir']
        # self.GT_NormData_path = self.data_setting['NormDataPath']
        self.resume_enable = self.train_setting['resume']
        self.lr = self.train_setting['lr']
        self.scale = self.data_setting['scale']

        self.Log_path = os.path.join(self.log_base_dir,'Log')
        self.Results_path = os.path.join(self.log_base_dir,'Results')
        self.ResultsRaw_path = os.path.join(self.log_base_dir,'ResultsRaw')
        self.Latent_path = os.path.join(self.log_base_dir,'Latent')
        
        #init logger and model
        self.logger = logger.getLogger("trainLogger")
        self.inflogger = logger.getLogger("easylogger")
        self.model = model.to(device)
        self.latentTable = latentTable.to(device)

        
        self.dataset = MultiVarDataset(args)
        self.dataset.ReadData()
        self.vars = self.dataset.dataset_varList
        self.unsupervised = self.data_setting['unsupervised']
        

        #optimizer and loss init
        self.lr = self.train_setting['lr']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_setting['lr'],betas=(0.9,0.999),weight_decay=1e-6)
        self.LatentOptimizer = optim.Adam(self.latentTable.parameters(), lr=self.train_setting['latent_lr'],betas=(0.9,0.999),weight_decay=1e-6)
        
        self.criterion = nn.MSELoss()
        #when init dataloader, must follow this order: TrainLoader --> InfLoader --> datasetInfo
        self.train_loader = self.dataset.getMultiVarTrainLoader()
        self.test_loader = self.dataset.getMultiVarInfLoader()
        self.datasetInfo = self.dataset.datasetInfo

        #some variables used to keep track the training process
        self.itera = 1
        if self.resume_enable:
            self.logger.info(f"Resume training from checkpoints...")
            self.resumeCheckPoint()
        
        #---time measurement variable---#
        self.model_time = 0
        self.train_cuda_time = 0
        self.total_time = 0


    
    def train_one_iteration(self,data):#inps -> [time,x,y,z]
        input_values,GT = data[0],data[1]
        input_values = input_values.to(device)
        GT = GT.to(device)
        self.optimizer.zero_grad()
        self.LatentOptimizer.zero_grad()
        # #*======calculate the original scaler value and its loss with GT=======#
        query_index = input_values[:,0].long()
        coords = input_values[:,1:]
        latent = self.latentTable(query_index)
        v_pred_R = self.model(coords,latent)
        reconstruction_loss = self.criterion(v_pred_R,GT)
        loss = reconstruction_loss
        loss.backward()
        self.LatentOptimizer.step()
        self.optimizer.step()
        return reconstruction_loss.detach().item()

    
    def trainNet(self,reg=0.001):
        reg = reg
        
        for itera in range(self.itera,self.num_epochs+1):
            reconstruction_loss = 0
            kl_loss = 0
            idx = 0
            tic = time.time()
            print('======='+str(itera)+'========')
            
            if (itera%100 == 0):
                self.setDecoderCheckPoint(itera)
                self.quickInf(itera)
            
            if (itera%self.checkpoints_interval == 0) or (itera==1):
                self.setDecoderCheckPoint(itera)
                
            
            for batch_idx,data in enumerate(self.train_loader):
                idx += 1
                reconstruction_loss_inc  = self.train_one_iteration(data)
                reconstruction_loss += reconstruction_loss_inc

            self.LatentOptimizer.zero_grad()
            kl_loss = self.latentTable.kl_loss()*reg
            kl_loss.backward()
            self.LatentOptimizer.step()
        
            toc = time.time()
            self.logger.info(f"Epochs {itera}: reconstructionLoss = {reconstruction_loss/idx} klLoss = {kl_loss} Time = {toc-tic}s lr = {self.optimizer.param_groups[0]['lr']} reg = {reg}")


    def quickInf(self,itera):
        for i in range(len(self.vars)):
            results_path = os.path.join(self.Results_path,self.vars[i])
            delFilesInDir(results_path)        
        with torch.no_grad():
            for embedding_i, var in enumerate(self.vars):
                t_index = 0
                for t in tqdm(self.test_loader[var]['t'],disable=False,desc=f"inferece variable {var}"):
                    t_index = t_index + 1 #the time slot starts from 1
                    coords = self.test_loader[var]['coords']
                    t_col = t*np.ones((coords.shape[0],1))
                    t_coords = np.concatenate((t_col,coords),axis=1)
                    embedding_index = embedding_i*np.ones((t_coords.shape[0],1)).astype(np.float32)
                    t_coords = np.concatenate((embedding_index,t_coords),axis=1)
                    data_loader = DataLoader(dataset = torch.FloatTensor(t_coords),batch_size=self.data_setting['batch_size'],shuffle=False,num_workers=8)
                    v_res = []
                    for coord in data_loader:
                        coord = coord.to(device)
                        with torch.no_grad():
                            latent = self.latentTable(coord[:,0].long(),train=False)
                            v_pred = self.model(coord[:,1:],latent)
                            v_res += list(v_pred.detach().cpu().numpy())
                    
                    v_res = np.array(v_res).transpose()
                    
                    v_res_total = self.concatVolumes(v_res,self.datasetInfo[var]['reverseBuffer'])
                    v_res_total = np.asarray(v_res_total,dtype='<f')
                    
                    save_file_name = f'STSR-INR-{var}-{t_index:04d}.raw'
                    save_file_path = os.path.join(self.Results_path,var,save_file_name)
                    v_res_total.tofile(save_file_path,format='<f')
        if not self.unsupervised:            
            total_PSNR = 0
            for k in range(len(self.vars)):
                results_path = os.path.join(self.Results_path,self.vars[k])
                gt_norm_path = os.path.dirname(self.datasetInfo[self.vars[k]]['data_path'])
                evalWidget = EvalMetric(gt_norm_path,results_path,verbose=True)
                mean_PSNR,PSNR_array = evalWidget.getPSNR()
                total_PSNR += mean_PSNR
                for i in range(1, len(PSNR_array)+1):
                    self.inflogger.info(f"Inf Result at Epoch {itera} for variable {self.vars[k]} at time index {i}: PSNR = {PSNR_array[i-1]} dB")
                self.inflogger.info(f"Inf Result for variable {self.vars[k]} at Epoch {itera}: {mean_PSNR} dB")
            self.inflogger.info(f"Total Averaged Inf Result at Epoch {itera}: {total_PSNR/len(self.vars)} dB\n")
    
    def setDecoderCheckPoint(self,itera):
        # save_fileName = f"{self.model_name}"+"".join([f"-{var}" for var in self.vars])+f"-{self.model_setting['init']}init-{self.model_setting['num_res']}res-{itera}.pth"
        save_fileName = f"{self.model_name}"+"".join([f"-{var}" for var in self.vars])+f"-{itera}.pth"
        save_content = {
            'model_name':self.model_name,
            'epoch':itera,
            'model_state_dict':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'cur_lr': self.lr,
            'datasetInfo':self.datasetInfo
        }
        save_file_path = os.path.join(self.Log_path,save_fileName)
        torch.save(save_content,save_file_path)
        
        latent_fileName ="Latent-"+"".join([f"-{var}" for var in self.vars])+f"-{itera}.pth"
        latent_save_content = {
            'latent_state_dict':self.latentTable.state_dict(),
            'latent_optimizer':self.LatentOptimizer.state_dict(),
        }
        save_file_path = os.path.join(self.Latent_path,latent_fileName)
        torch.save(latent_save_content,save_file_path)

    
    def resumeCheckPoint(self):
        lateset_model_path = getLatestModelPath(self.Log_path)
        saved_content = torch.load(lateset_model_path,map_location='cpu')
        self.itera = saved_content['epoch']
        self.model.load_state_dict(saved_content['model_state_dict'])
        self.optimizer.load_state_dict(saved_content['optimizer'])
        
        lateset_latent_path = getLatestModelPath(self.Latent_path)
        latent_saved_content = torch.load(lateset_latent_path,map_location='cpu')
        self.latentTable.load_state_dict(latent_saved_content['latent_state_dict'])
        self.LatentOptimizer.load_state_dict(latent_saved_content['latent_optimizer'])
    
    def concatVolumes(self,split_data,reverseBuffer):
        dims,Reverse_split_axis_order,splitTimes,complemtAxis = reverseBuffer['HR_split_dims'], reverseBuffer['split_axis_order'],reverseBuffer['splitTimes'],reverseBuffer['complemtAxis']
        split_data = [data.reshape(dims[2],dims[1],dims[0]).transpose() for data in split_data]
        # for i in range(splitTimes*3): #* Remeber to change with splitvolumes
        for i in range(splitTimes):
            splitDataLsSize = len(split_data)
            split_data = [np.concatenate((split_data[idx],split_data[idx+1]),axis=Reverse_split_axis_order[i]) for idx in range(0,splitDataLsSize,2)]
            if(complemtAxis[i]!=-1):
                split_data = [reverseComplementZero(data,complemtAxis[i],self.scale) for data in split_data]
        res = split_data[0].flatten('F')
        return res
    

if __name__ == "__main__":
    evalWidget = EvalMetric(r'D:\VSCodeLib\data\vortsNorm',r'D:\VSCodeLib\STCoordNet\Exp\Baseline\Results',verbose=True)
    print(str(evalWidget.getPSNR())+' dB')