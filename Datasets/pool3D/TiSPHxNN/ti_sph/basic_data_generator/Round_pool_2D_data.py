import taichi as ti
import numpy as np

# from .Data_generator import Data_generator
from .Data_generator import Data_generator

@ti.data_oriented
class Round_pool_2D_data(Data_generator):
    def __init__(self, r_hollow: ti.f32, r_in: ti.f32, r_out:ti.f32, angular_vel:ti.f32, span: ti.f32):
        self.r_in = r_in
        self.r_out = r_out
        self.angular_vel = angular_vel
        self.outer_radius = r_out / 2
        self.inner_radius = r_in / 2
        self.hollow_radius = r_hollow / 2
        self.span = span

        # Create a grid
        self.grid_x, self.grid_y = np.mgrid[-self.outer_radius : self.outer_radius : self.span, -self.outer_radius : self.outer_radius : self.span]

        # Carve out the circle
        self.squared_distances = self.grid_x**2 + self.grid_y**2
        self.fluid_mask = (self.squared_distances <= self.inner_radius**2) & (self.squared_distances > self.hollow_radius)
        self.bound_mask = (self.squared_distances > self.inner_radius**2) & (self.squared_distances <= self.outer_radius**2)

        self.fluid_particle_x = self.grid_x[self.fluid_mask]
        self.fluid_particle_y = self.grid_y[self.fluid_mask]
        self.fluid_part_num = self.fluid_particle_x.shape[0]
        self.fluid_part_pos = np.stack((self.fluid_particle_x.reshape(-1), self.fluid_particle_y.reshape(-1)), -1)

        ''' vorticity free '''
        distances = np.sqrt(self.fluid_part_pos[:, 0]**2 + self.fluid_part_pos[:, 1]**2)
        omega_r = angular_vel / (distances**2)
        self.fluid_part_vel_x = -omega_r * self.fluid_particle_y
        self.fluid_part_vel_y = omega_r * self.fluid_particle_x
        ''' rigid body rotation '''
        # self.fluid_part_vel_x = -self.angular_vel * self.fluid_particle_y
        # self.fluid_part_vel_y = self.angular_vel * self.fluid_particle_x

        self.fluid_part_vel = np.stack((self.fluid_part_vel_x.reshape(-1), self.fluid_part_vel_y.reshape(-1)), -1)

        self.bound_particle_x = self.grid_x[self.bound_mask]
        self.bound_particle_y = self.grid_y[self.bound_mask]
        self.bound_part_num = self.bound_particle_x.shape[0]
        self.bound_part_pos = np.stack((self.bound_particle_x.reshape(-1), self.bound_particle_y.reshape(-1)), -1)
