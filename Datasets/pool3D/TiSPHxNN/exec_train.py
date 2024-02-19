print('new network type 01')
import os

from Autoencoders.ConvAE_velocity import *
from Dataset_processing import *

strategy_list = ['skip_bottleneck', 'whole']
model_path = os.path.join('./model')
model_file_list = [None, os.path.join(model_path,'epochs_699.pth')]
main_folder_path = './dataset_train'

if_freeze_parameters = False
if_crop = True
strategy = strategy_list[1]
model_file = model_file_list[0]
submodule_type = 1

exclude_threshold = None

res = 256

attr_name_1 = 'velocity'
attr_name_2 = 'strainRate2vorticity'
attr_name_3 = 'density'
platform = 'cuda'

'''
Training process:
'''
output_path = os.path.join(main_folder_path, 'test_output')

dataset_file_path_1 = os.path.join(main_folder_path, 'dataset')
dataset_file_path_2 = os.path.join(main_folder_path, 'dataset')
dataset_file_path_3 = os.path.join(main_folder_path, 'dataset')

model = TrainConvAutoencoder(res, attr_name_1, dataset_file_path_1, 
                                  attr_name_2, dataset_file_path_2, 
                                  attr_name_3, dataset_file_path_3, 
                                  platform, False, lr=1e-4, submodule_type=submodule_type)

# model.train_velocityBased (num_epochs=8000, network_model_path=model_path,  former_model_file_path=None)
# model.output_bottleneck(   model_file_path='./model/epochs_299.pth', output_path=output_path)
model.train(num_epochs=40000, crop=if_crop,
            network_model_path=model_path, strategy=strategy, former_model_file_path=model_file, save_step=50, freeze_param=if_freeze_parameters, exclude_threshold=exclude_threshold)
# model.train_histBased(     num_epochs=8000, network_model_path=model_path, former_model_file_path=None)

# datavis_1darray(output_path, output_path, 'bottleneck', 0, 659)