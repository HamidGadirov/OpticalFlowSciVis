import argparse
from utils import *
from model import *
from dataio import *
from train import *
import os
import time
from torchsummary import summary


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

p = argparse.ArgumentParser()#multiVarCoord
p.add_argument('--config_path', type=str, default='./tasks/configs/CylinderVelM.yml', help='The path of the config file') #diffusion
opt = p.parse_args()

def setup_Exp(settings_path=None):
    """setup the settings and logger, create Exp files
    """
    #parse the settings
    settings_path = opt.config_path if settings_path == None else settings_path
    args = yaml_loader(settings_path)
    mode = args['mode']
    model_setting = args['model_setting']
    train_setting = args['train_setting']
    data_setting = args['data_setting']
    version =f"v{train_setting['version']}"

    # add some useful attribute to args
    log_base_dirName = version# getYMD() + version
    args['log_base_dir'] = os.path.join(train_setting['log_root_dir'],log_base_dirName)

    #create log base dir with version tag
    ensure_dirs(args['log_base_dir'])
    for sub_dirs in ['Log','Results','ResultsRaw','Latent']:
        ensure_dirs(os.path.join(args['log_base_dir'],sub_dirs))
        if (sub_dirs == "Results" or sub_dirs == "ResultsRaw"):
            for var in data_setting['dataset']:
                ensure_dirs(os.path.join(args['log_base_dir'],sub_dirs,var))
    
    #move the model,setting and main file into base_dir
    copy_modelSetting(args['log_base_dir']) #move everything to log
    _, setting_file_name = os.path.split(settings_path)
    setting_savedTo = os.path.join(args['log_base_dir'],setting_file_name)
    if os.path.exists(setting_savedTo):
        os.remove(setting_savedTo)
    shutil.copy(opt.config_path,setting_savedTo)

    #setup logger
    logger = setup_logger(log_file_dir=os.path.join(args['log_base_dir'],'Log'))

    return args,train_setting,model_setting,data_setting,logger

def inf(latent_path,model_path,sampled_timeSteps,dims,embedding_nums,Log_Dir,sampled_timeStepsIndex,isVAD):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args,train_setting,model_setting,data_setting,logger = setup_Exp()
    mode = args['mode']
    vars_num = len(data_setting['dataset'])
    splitTimes = data_setting['splitTimes']
    output_nums = 1
    
    interpolation_factor = 20
    
    #--------Model init------------------#
    Decoder = STSR_INR(in_coords_dims=4,out_features=output_nums,num_res=model_setting['num_res'],embedding_dims=256,outermost_linear=False)
    latentTable = VarVADEmbedding(embedding_dims=256,embedding_nums=embedding_nums)
    #-----------------------------------#
    
    model_saved_content = torch.load(model_path,map_location='cpu')
    latent_saved_content = torch.load(latent_path,map_location='cpu')
    Decoder.load_state_dict(model_saved_content['model_state_dict'])
    latentTable.load_state_dict(latent_saved_content['latent_state_dict'])
    embeddingTable = torch.zeros((embedding_nums*interpolation_factor+1,256))
    if not isVAD:
        trainedEmbeddingTable = latentTable.embedding.weight
    else:
        trainedEmbeddingTable = latentTable.weight_mu
    key_indexs = []
    Decoder.to(device)
    with torch.no_grad():
        for i in range(trainedEmbeddingTable.shape[0]):
            key_indexs.append(i*(interpolation_factor+1)+1)
            embeddingTable[i*(interpolation_factor+1)] = trainedEmbeddingTable[i]
        embeddingTable=interpolation_embedding(embeddingTable,key_indexs)
        coords = get_mgrid([dims[0],dims[1],dims[2]],dim=3,s=1)
        input_coords = []
        for t_idx,t in enumerate(sampled_timeSteps):
            print(t_idx)
            t_col = t*np.ones((coords.shape[0],1))
            coord_loader = DataLoader(dataset = torch.FloatTensor(np.concatenate((t_col,coords),axis=1)),batch_size=32000*4,shuffle=False)
            for latent_idx,latentVec in enumerate(embeddingTable):
                v_res = []
                latentVec = latentVec.to(device)
                for coord in coord_loader:
                    coord = coord.to(device)
                    v_pred = Decoder(coord,latentVec)
                    v_res += list(v_pred.detach().cpu().numpy())
                v_res = np.array(v_res).transpose()
                
                v_res = np.asarray(v_res,dtype='<f')
                
                ensure_dirs(os.path.join(Log_Dir,f"Time{sampled_timeStepsIndex[t_idx]+1}"))
                save_file_name = f'Latent{latent_idx+1:04d}.dat'
                save_file_path = os.path.join(Log_Dir,f"Time{sampled_timeStepsIndex[t_idx]+1}",save_file_name)
                v_res.tofile(save_file_path,format='<f')

if __name__ == "__main__":
    VADlatent_path = ... # todo:  trained ckpt for latent embedding
    VADmodel_path = ... # todo: trained ckpt for decoder weights
    VADLog_Dir = ... # todo: specify the log dir for result
    
    dims = [640,240,80]
    total_samples = 50
    embedding_nums = 4
    total_timeSteps = [i for i in range(1,50+1)]
    sampled_timeStepsIndex = [49]
    sampled_timeSteps = []
    for idx, t in enumerate(total_timeSteps):
        if idx in sampled_timeStepsIndex:
            t_norm = (2*(t - 1)/(total_samples - 1) -1)
            sampled_timeSteps.append(t_norm)
    inf(embedding_nums=embedding_nums,latent_path=VADlatent_path,model_path=VADmodel_path,sampled_timeSteps=sampled_timeSteps,dims=dims,Log_Dir=VADLog_Dir,sampled_timeStepsIndex=sampled_timeStepsIndex,isVAD=True)
