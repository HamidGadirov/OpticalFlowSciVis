import taichi as ti
import numpy as np

# from .Data_generator import Data_generator

@ti.data_oriented
class Squared_pool_3D_data():
    def __init__(self, container_height: ti.f32, container_size: ti.f32, fluid_height: ti.f32, span: ti.f32, layer:ti.i32):

        self.container_height = container_height
        self.container_size = container_size
        self.fluid_height = fluid_height
        self.span = span
        self.fluid_empty_height = self.fluid_height - self.container_height/2

        self.grid_x, self.grid_y, self.grid_z = \
            np.mgrid[\
                -self.container_size/2 : self.container_size/2 : self.span, \
                -self.container_height/2 : self.container_height/2 : self.span,\
                -self.container_size/2 : self.container_size/2 : self.span,]
        
        self.mask_inner_space =  (self.grid_x > -self.container_size/2 + self.span*layer) & (self.grid_x < self.container_size/2 - self.span*layer) & \
                            (self.grid_y > -self.container_height/2 + self.span*layer) & (self.grid_y < self.container_height/2 - self.span*layer) & \
                            (self.grid_z > -self.container_size/2 + self.span*layer) & (self.grid_z < self.container_size/2 - self.span*layer)
        
        self.mask_bound = ~self.mask_inner_space
        self.mask_fluid = self.mask_inner_space & (self.grid_y < self.fluid_empty_height) \
            # & (self.grid_x < 0)

        self.fluid_position_x = self.grid_x[self.mask_fluid]
        self.fluid_position_y = self.grid_y[self.mask_fluid]
        self.fluid_position_z = self.grid_z[self.mask_fluid]
        self.fluid_part_pos = np.stack((self.fluid_position_x.reshape(-1), self.fluid_position_y.reshape(-1), self.fluid_position_z.reshape(-1)), -1)
        self.fluid_part_num = self.fluid_part_pos.shape[0]

        self.bound_position_x = self.grid_x[self.mask_bound]
        self.bound_position_y = self.grid_y[self.mask_bound]
        self.bound_position_z = self.grid_z[self.mask_bound]
        self.bound_part_pos = np.stack((self.bound_position_x.reshape(-1), self.bound_position_y.reshape(-1), self.bound_position_z.reshape(-1)), -1)
        self.bound_part_num = self.bound_part_pos.shape[0]
