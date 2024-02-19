import taichi as ti
import numpy as np

from .Data_generator import *
from ..basic_obj.Obj_Particle import Particle

@ti.data_oriented
class Cube_generator(Data_generator):

    num = 0

    def __init__(self, obj: Particle, lb: np.array, rt: np.array):
        self.obj = obj
        self.lb = lb
        self.rt = rt
        self.dim = len(lb)

    def __init__(self, obj: Particle, lb: ti.Vector, rt: ti.Vector):
        self.obj = obj
        self.lb = lb.to_numpy()
        self.rt = rt.to_numpy()
        self.dim = len(lb)

    def push_pos(
            self, 
            factor: float = 1.0,
            span: float = -1,
            ):
        
        temp_span = factor * self.obj.getObjPartSize()
        if not span < 0:
            temp_span = span
            
        self.generate_pos_based_on_span(temp_span)
        cube_pos_data = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.num)
        cube_pos_data.from_numpy(self.np_pos)
        self.ker_push_pos(cube_pos_data, self.obj.pos, self.obj.tiGetObjStackTop())
        return self.num
    
    def pushed_num_preview(self, factor: float = 1.0, span: float = -1):
        temp_span = factor * self.obj.getObjPartSize()
        if not span < 0:
            temp_span = span
            
        self.generate_pos_based_on_span(temp_span)
        return self.num

    def generate_pos_based_on_span(self, span: float):
        self.voxel_shape = np.ceil((self.rt - self.lb) / span)
        self.voxel_shape = self.voxel_shape.astype(np.int32)
        # print("DEBUG: voxel_shape: ", voxel_shape)

        pos_frac = []
        index_frac = []
        for i in range(self.dim):
            pos_frac.append(np.linspace(self.lb[i], self.lb[i]+span*self.voxel_shape[i], self.voxel_shape[i]+1))
            index_frac.append(np.linspace(0,self.voxel_shape[i], self.voxel_shape[i]+1).astype(np.int32))

        self.np_pos = np.array(np.meshgrid(*pos_frac)).T.reshape(-1, self.dim)
        self.np_index = np.array(np.meshgrid(*index_frac)).T.reshape(-1, self.dim)

        self.num = self.np_pos.shape[0]

        return (self.np_index, self.np_pos, self.num)

    def _get_index(self, to):
        return to.from_numpy(self.np_index)

    def get_shape(self):
        return self.voxel_shape

    @ti.kernel
    def ker_push_pos(self, cube_pos_data:ti.template(), pos_arr:ti.template(), stack_top:ti.i32):
        for i in range(self.num):
            pos_arr[stack_top+i] = cube_pos_data[i]


        
