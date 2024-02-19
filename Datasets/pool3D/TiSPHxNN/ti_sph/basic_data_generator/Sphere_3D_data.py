import taichi as ti
import numpy as np

@ti.data_oriented
class Sphere_3D_data():
    def __init__(self, radius: ti.f32, pos: ti.types.vector(3, ti.f32), span: ti.f32):

        self.radius = radius
        self.pos = pos
        self.span = span

        self.grid_x, self.grid_y, self.grid_z = \
            np.mgrid[\
                -self.radius : self.radius : self.span, \
                -self.radius : self.radius : self.span, \
                -self.radius : self.radius : self.span, ]
        
        self.mask_fluid =  (self.grid_x**2 + self.grid_y**2 + self.grid_z**2) < self.radius**2

        self.fluid_position_x = self.grid_x[self.mask_fluid]
        self.fluid_position_y = self.grid_y[self.mask_fluid]
        self.fluid_position_z = self.grid_z[self.mask_fluid]
        self.fluid_part_pos = np.stack((self.fluid_position_x.reshape(-1), self.fluid_position_y.reshape(-1), self.fluid_position_z.reshape(-1)), -1)
        self.fluid_part_num = self.fluid_part_pos.shape[0]
