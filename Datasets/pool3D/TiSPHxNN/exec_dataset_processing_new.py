import taichi as ti
import ti_sph as tsph
from Dataset_processing import *

ti.init(arch=ti.gpu)

main_path = '../output'
output_path = '../organized_output'
input_compressed = True
output_compressed = True
attr_list = ['vel', 'pos', 'sensed_density', 'strainRate']
output_attr_list = ['velocity', 'position', 'density', 'strainRate']

for attr, output_attr in zip(attr_list, output_attr_list):
    tsph.SeqData(path=main_path, attr_name=attr, compressed=input_compressed)\
        .reshape_to_3d(index_attr_name='node_index', output_path=output_path, output_name=output_attr, compressed=output_compressed)