import argparse
from utils import *
from model import *
from dataio import *
from train import *
import os
import time
from torchsummary import summary
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def setup_Exp(opt,settings_path=None,version=None):
    """setup the settings and logger, create Exp files
    """
    #parse the settings
    settings_path = opt.config_path if settings_path == None else settings_path
    # print("settings_path", settings_path)
    args = yaml_loader(settings_path)
    mode = args['mode']
    model_setting = args['model_setting']
    train_setting = args['train_setting']
    data_setting = args['data_setting']
    # train_setting['version'] = "nyx" ####
    print("train_setting['version']", train_setting['version'])
    # print("version", version)
    train_setting['version'] = train_setting['version'] if version == None else version
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

def main(opt):
	args,train_setting,model_setting,data_setting,logger = setup_Exp(opt)
	mode = args['mode']
	vars_num = len(data_setting['dataset'])
	splitTimes = data_setting['splitTimes']
	init_feat = model_setting['init']
	omega_0 = model_setting['omega_0']
	# output_nums = 2**(splitTimes*3)
	output_nums = 2**(splitTimes)

	Decoder = STSR_INR(in_coords_dims=4,out_features=output_nums,init_features=init_feat,num_res=model_setting['num_res'],embedding_dims=256,outermost_linear=True,omega_0=omega_0)
	latentTable = VarVADEmbedding(embedding_dims=256,embedding_nums=vars_num)

	t = Train(Decoder,latentTable,args,logger)
	t.trainNet()

if __name__ == "__main__":
    # """
    p = argparse.ArgumentParser()
    # p.add_argument('--config_path', type=str, default='./tasks/configs/CylinderVelM.yml', help='The path of the config file') 
    # p.add_argument('--config_path', type=str, default='./configs/nyx_train.yml', help='The path of the config file')
    p.add_argument('--config_path', type=str, default='./configs/castro_train.yml', help='The path of the config file')
    opt = p.parse_args()
    print("start")
    start_time = time.time()
    main(opt)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Inference time: {elapsed_time:.2f} seconds") 
    print("end")
    # """

    # GT_filePath = '../Datasets/Nyx/nyx_ensemble_test/Nyx/Exec/LyA/nyx3d_skip5_128_log10norm_OmM_01375_OmB_00225_h_055_STSRINR_train.pkl'
    # eval_dirPath = './Exp/nyx/vnyx/Results/nyx_Scalar/'

    GT_filePath = '../Datasets/Castro/Exec/science/wdmerger/castro3d_128_log10norm_massP_090_massS_085_test_params.pkl'
    eval_dirPath = './Exp/castro/vcastro/Results/castro_Scalar/'

    metrics_calculator = MetricsCalculator(GT_filePath, eval_dirPath, verbose=True)
    mean_psnr, psnr_list = metrics_calculator.getPSNR()
    print("Mean PSNR:", mean_psnr)

# python3 main.py --config_path './configs/nyx_train.yml'
# python3 main.py --config_path './configs/nyx_inf.yml'

# inference is slow, much slower than LREST

