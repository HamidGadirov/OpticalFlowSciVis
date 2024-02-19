import taichi as ti
import numpy as np

from .Data_generator import Data_generator

DEFAULT = None
@ti.data_oriented
class Cube_data(Data_generator):
    FIXED_CELL_SIZE = 0
    FIXED_GRID_RES = 1
    def __init__(self, 
                 span: float, type,
                 lb: ti.Vector = DEFAULT, rt: ti.Vector = DEFAULT, # These parameters are used for the type FIXED_CELL_SIZE
                 grid_res: ti.Vector = DEFAULT, grid_center: ti.Vector = DEFAULT, # These parameters are used for the type FIXED_GRID_RES
                 ):
        print("creating Cube_data ...")

        # Input parameters
        self.lb = lb
        self.rt = rt
        self.grid_res = grid_res
        self.grid_center = grid_center
        self.span = span
        self.type = type

        # Derived parameters
        if self.type == self.FIXED_GRID_RES:
            self.dim = len(grid_res)
        elif self.type == self.FIXED_CELL_SIZE:
            self.dim = len(lb)
        if self.type == self.FIXED_GRID_RES: # for the type FIXED_GRID_RES, some operations are needed to get the lb and rt to reduce the problem to FIXED_CELL_SIZE
            temp_grid_size = self.grid_res * self.span
            self.lb = self.grid_center - temp_grid_size / 2
            self.rt = self.grid_center + temp_grid_size / 2
        self.shape = np.ceil((self.rt - self.lb) / span).astype(np.int32) # the number of voxels in each dimension

        # Output data
        self.pos = None
        self.num = None

        # Generate data
        pos_frac = [] # the position of each voxel in each dimension
        index_frac = [] # corresponding index of elements in pos_frac 
        for i in range(self.dim):
            pos_frac.append(np.linspace(self.lb[i], self.lb[i]+span*self.shape[i], self.shape[i]+1))
            index_frac.append(np.linspace(0,self.shape[i], self.shape[i]+1).astype(np.int32))
        # returned values
        self.pos = np.array(np.meshgrid(*pos_frac)).T.reshape(-1, self.dim) # the pos array takes the form of (num, dim)
        self.index = np.array(np.meshgrid(*index_frac)).T.reshape(-1, self.dim) # the index array takes the form of (num, dim)
        self.num = self.pos.shape[0]
    
        print('Done!')

    def translate(self, offset: ti.Vector):
        self.pos += offset.to_numpy()
        return self
