from Dataset_processing import *

ti.init(arch=ti.gpu)


'''
Training dataset generation
'''
main_path = './dataset_train'
rawdata_folder = 'rawdata'
dataset_folder = 'dataset'
datavis_folder = 'datavis'
clear_folder(os.path.join(main_path, rawdata_folder))
clear_folder(os.path.join(main_path, dataset_folder))
clear_folder(os.path.join(main_path, datavis_folder))

operation_list = [
    'flipud', 'fliplr', 'transpose', 'flipud_fliplr'
    ]
length = len(operation_list)+1

number_of_frames = concatDataset('./',
                [
                    'raw_t1', 
                    'raw_t2', 
                    # 'raw_t3', 
                    'raw_t4', 
                    'raw_t5', 
                    'raw_t6',
                    'raw_t7',
                    'raw_t8',
                    'raw_t9',
                    'raw_t10',
                    # 'raw_t11',
                    # 'raw_t12',
                    # 'raw_t13',
                 ],
                ['node_index', 'vel', 'pos', 'sensed_density', 'strainRate'], 
                os.path.join(main_path, rawdata_folder))
gridExport_density(       os.path.join(main_path, rawdata_folder), os.path.join(main_path, dataset_folder), 0, number_of_frames, operations=operation_list)
gridExport_vel(           os.path.join(main_path, rawdata_folder), os.path.join(main_path, dataset_folder), 0, number_of_frames, operations=operation_list)
gridExport_strainRate(    os.path.join(main_path, rawdata_folder), os.path.join(main_path, dataset_folder), 0, number_of_frames, operations=operation_list)
process_strainRate_to(    os.path.join(main_path, dataset_folder), os.path.join(main_path, dataset_folder), 0, number_of_frames*length, to='vorticity', use_density_mask=True)
process_vel_to_strainRate(os.path.join(main_path, dataset_folder), os.path.join(main_path, dataset_folder), 0, number_of_frames*length, 7.0/258, True, further_to='vorticity')


# scivis_R2toR1(os.path.join(main_path, dataset_folder), os.path.join(main_path, datavis_folder), 0, number_of_frames*length, 'density', stride=1)
# scivis_R2toR2(os.path.join(main_path, dataset_folder), os.path.join(main_path, datavis_folder), 0, number_of_frames*length, 'velocity', channel_at_end=True)
# scivis_R2toR1(os.path.join(main_path, dataset_folder), os.path.join(main_path, datavis_folder), 0, number_of_frames*length, 'strainRate2vorticity')
# scivis_R2toR1(os.path.join(main_path, dataset_folder), os.path.join(main_path, datavis_folder), 0, number_of_frames*length, 'vel2vorticity')

'''
Testing dataset generation
'''
# main_path = './dataset_test'
# rawdata_folder = 'rawdata'
# dataset_folder = 'dataset'
# datavis_folder = 'datavis'
# clear_folder(os.path.join(main_path, rawdata_folder))
# clear_folder(os.path.join(main_path, dataset_folder))
# clear_folder(os.path.join(main_path, datavis_folder))

# operation_list = []
# length = len(operation_list)+1

# number_of_frames = concatDataset('.\\',
#                 ['raw_t3'],
#                 ['node_index', 'vel', 'pos', 'sensed_density', 'strainRate'], 
#                 os.path.join(main_path, rawdata_folder))
# gridExport_density(       os.path.join(main_path, rawdata_folder), os.path.join(main_path, dataset_folder), 0, number_of_frames, operations=operation_list)
# gridExport_vel(           os.path.join(main_path, rawdata_folder), os.path.join(main_path, dataset_folder), 0, number_of_frames, operations=operation_list)
# gridExport_strainRate(    os.path.join(main_path, rawdata_folder), os.path.join(main_path, dataset_folder), 0, number_of_frames, operations=operation_list)
# process_strainRate_to(    os.path.join(main_path, dataset_folder), os.path.join(main_path, dataset_folder), 0, number_of_frames*length, to='vorticity', use_density_mask=True)
# process_vel_to_strainRate(os.path.join(main_path, dataset_folder), os.path.join(main_path, dataset_folder), 0, number_of_frames*length, 7.0/258, True, further_to='vorticity')
# scivis_R2toR1(os.path.join(main_path, dataset_folder), os.path.join(main_path, datavis_folder), 0, number_of_frames*length, 'density')
# scivis_R2toR1(os.path.join(main_path, dataset_folder), os.path.join(main_path, datavis_folder), 0, number_of_frames*length, 'strainRate2vorticity')
# scivis_R2toR1(os.path.join(main_path, dataset_folder), os.path.join(main_path, datavis_folder), 0, number_of_frames*length, 'vel2vorticity')


# clear_folder(raw_path)
# clear_folder(raw_test_path)
# clear_folder(raw_train_path)
# clear_folder(organized_path)
# clear_folder(organized_test_path)
# clear_folder(organized_train_path)

# clear_folder(sciVis_path)
# clear_folder(dataVis_path)
# clear_folder(hist_path)

# concatDataset('\\Users\\xuyan\\Documents\\GitHub\\output', 
#               ['raw_t1', 'raw_t2', 'raw_t3'],
#               ['node_index', 'vel', 'pos', 'sensed_density', 'strainRate'], 
#               raw_path)

# concatDataset('\\Users\\xuyan\\Documents\\GitHub\\output',
#                 ['raw_t3'],
#                 ['node_index', 'vel', 'pos', 'sensed_density', 'strainRate'], 
#                 raw_test_path)

# start_index = 0
# end_index = 984
# operation_list = ['flipud', 'fliplr', 'transpose', 'flipud_fliplr']
# organized_path = './output/organized/'
# organized_vis_path = './output/organized_vis/'
# length = len(operation_list)+1
# gridExport_density(raw_path, organized_path, start_index, end_index, operations=operation_list)
# gridExport_vel(raw_path, organized_path, start_index, end_index, operations=operation_list)
# gridExport_strainRate(raw_path, organized_path, start_index, end_index, operations=operation_list)
# process_strainRate_to(organized_path, organized_path, start_index, end_index*length, to='vorticity', use_density_mask=True)
# process_vel_to_strainRate(organized_path, organized_path, start_index, end_index*length, 7.0/258, True, further_to='vorticity')
# process_minus(organized_path, organized_path, start_index, end_index*length, 'vel2vorticity', 'strainRate2vorticity')
# scivis_R2toR1(organized_path, organized_vis_path, start_index, end_index*length, 'density')
# scivis_R2toR1(organized_path, organized_vis_path, start_index, end_index*length, 'strainRate2vorticity')
# scivis_R2toR1(organized_path, organized_vis_path, start_index, end_index*length, 'vel2vorticity')
# scivis_R2toR1(organized_path, organized_vis_path, start_index, end_index*length, 'vel2vorticityMINUSstrainRate2vorticity')

# start_index = 0
# end_index = 798
# gridExport_density(raw_train_path, organized_train_path, start_index, end_index)
# gridExport_vel(raw_train_path, organized_train_path, start_index, end_index)
# gridExport_strainRate(raw_train_path, organized_train_path, start_index, end_index)
# process_strainRate_to(organized_train_path, organized_train_path, start_index, end_index, to='vorticity', use_density_mask=True)
# process_vel_to_strainRate(organized_train_path, organized_train_path, start_index, end_index, 7.0/258, True, further_to='vorticity')
# process_minus(organized_train_path, organized_train_path, start_index, end_index, 'vel2vorticity', 'strainRate2vorticity')

# start_index = 0
# end_index = 185
# gridExport_density(raw_test_path, organized_test_path, start_index, end_index)
# gridExport_vel(raw_test_path, organized_test_path, start_index, end_index)
# gridExport_strainRate(raw_test_path, organized_test_path, start_index, end_index)
# process_strainRate_to(organized_test_path, organized_test_path, start_index, end_index, to='vorticity', use_density_mask=True)
# process_vel_to_strainRate(organized_test_path, organized_test_path, start_index, end_index, 7.0/258, True, further_to='vorticity')
# process_minus(organized_test_path, organized_test_path, start_index, end_index, 'vel2vorticity', 'strainRate2vorticity')

# start_index = 0
# end_index = 798
# scivis_R2toR1(organized_train_path, organized_train_path, start_index, end_index, 'density')
# scivis_R2toR1(organized_train_path, organized_train_path, start_index, end_index, 'strainRate2vorticity')
# scivis_R2toR1(organized_train_path, organized_train_path, start_index, end_index, 'vel2vorticity')
# scivis_R2toR1(organized_train_path, organized_train_path, start_index, end_index, 'vel2vorticityMINUSstrainRate2vorticity')
# datavis_hist_R2toR1(organized_train_path, organized_train_path, start_index, end_index, 'strainRate2vorticity', -100, 100)
# datavis_hist_R2toR1(organized_train_path, organized_train_path, start_index, end_index, 'vel2vorticity', -100, 100)
# datavis_hist_R2toR1(organized_train_path, organized_train_path, start_index, end_index, 'vel2vorticityMINUSstrainRate2vorticity', -10, 10)

# process_abs(organized_path, organized_path, start_index, end_index, 'vel2vorticity')
# process_abs(organized_path, organized_path, start_index, end_index, 'strainRate2vorticity')
# process_abs(organized_path, organized_path, start_index, end_index, 'vel2vorticityMINUSstrainRate2vorticity')

# process_hist(organized_path, hist_path, start_index, end_index, 'abs_strainRate2vorticity', 0, 300)



# analyze_filenames_in_folder(raw_path, 'npy')
# analyze_filenames_in_folder(organized_path, 'npy')
# analyze_filenames_in_folder(sciVis_path, 'png')
# analyze_filenames_in_folder(dataVis_path, 'png')
# analyze_filenames_in_folder(hist_path, 'npy')



