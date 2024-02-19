import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

from .Data_generator import Data_generator

@ti.data_oriented
class Wing2412_data_2D(Data_generator):
    def __init__(self, span: ti.f32, chord_length: ti.f32, pos: ti.types.vector(2, ti.f32)):
        print("creating Wing2412_data_2D ...")

        #  Output data
        self.num = None
        self.pos = None

        # Input parameters
        self.span = span
        self.chord_length = chord_length
        self.shift_pos = pos

        # Derived parameters
        self.resolution = int(self.chord_length / self.span * 5)

        # Airfoil parameters
        self.m = 0.02  # Maximum camber
        self.p = 0.4   # Location of maximum camber
        self.t = 0.12  # Maximum thickness as a fraction of the chord
        self.c = self.chord_length   # Chord length
        self.x = np.linspace(0, self.c, self.resolution)  # x coordinates

        # Calculate camber line and thickness
        self.yc = self.camber_line(self.m, self.p, self.c, self.x)
        self.yt = self.thickness(self.t, self.c, self.x)

        # Combine camber and thickness to form the airfoil surfaces
        self.yu = self.yc + self.yt
        self.yl = self.yc - self.yt

        # Create a grid
        self.grid_x, self.grid_y = np.mgrid[0:self.c:self.span, -self.c:self.c:self.span]

        # Create a mask for points within the airfoil
        self.mask = (self.grid_y >= np.interp(self.grid_x, self.x, self.yl)) & (self.grid_y <= np.interp(self.grid_x, self.x, self.yu))

        # Extract particle locations
        self.particle_x = self.grid_x[self.mask]
        self.particle_y = self.grid_y[self.mask]

        # Shift the airfoil to the given position
        self.x += self.shift_pos[0]
        self.yu += self.shift_pos[1]
        self.yl += self.shift_pos[1]
        self.particle_x += self.shift_pos[0]
        self.particle_y += self.shift_pos[1]

        # Output data
        self.num = self.particle_x.shape[0]
        self.pos = np.stack((self.particle_x.reshape(-1), self.particle_y.reshape(-1)), -1)

        print('Done!')


    def camber_line(self, m:float, p:float, c:float, x:float):
        return np.where((x>=0)&(x<=(c*p)),
                        m * (x / np.power(p,2)) * (2.0 * p - (x / c)),
                        m * ((c - x) / np.power(1-p,2)) * (1.0 + (x / c) - 2.0 * p ))

    def thickness(self, t:float, c:float, x:float):
        term1 =  0.2969 * np.sqrt(x / c)
        term2 = -0.1260 * (x / c)
        term3 = -0.3516 * np.power(x / c, 2)
        term4 =  0.2843 * np.power(x / c, 3)
        term5 = -0.1015 * np.power(x / c, 4)
        return t * c * (term1 + term2 + term3 + term4 + term5) / 0.2
    
    def demo(self):
        # Plot airfoil
        plt.figure(figsize=(10, 5))
        plt.plot(self.x, self.yu, 'r', label='Upper Surface')
        plt.plot(self.x, self.yl, 'b', label='Lower Surface')
        plt.scatter(self.particle_x, self.particle_y, s=10, color='green', label='Particles')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('NACA 2412 Airfoil with Particles')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')  # Equal scaling
        plt.show()

@ti.data_oriented
class Wing2412_data_2D_with_cube(Wing2412_data_2D):
    def __init__(self, span: ti.f32, chord_length: ti.f32, pos: ti.types.vector(2, ti.f32), cube_lb: ti.types.vector(2, ti.f32), cube_rt: ti.types.vector(2, ti.f32), attack_angle: ti.f32=0.0):
        print("creating Wing2412_data_2D_with_cube ...")

        #  Output data
        self.wing_num = None
        self.wing_pos = None
        self.fluid_num = None
        self.fluid_pos = None

        # Input parameters
        self.span = span
        self.chord_length = chord_length
        self.shift_pos = pos
        self.cube_lb = cube_lb
        self.cube_rt = cube_rt
        self.bound_layer = 3
        self.vacum_layer = 0
        self.attack_angle = np.radians(-attack_angle)

        # Derived parameters
        self.resolution = int(self.chord_length / self.span * 5)
        self.rotation_matrix = np.array([[np.cos(self.attack_angle), -np.sin(self.attack_angle)], 
                            [np.sin(self.attack_angle), np.cos(self.attack_angle)]])

        # Airfoil parameters
        self.m = 0.02  # Maximum camber
        self.p = 0.4   # Location of maximum camber
        self.t = 0.12  # Maximum thickness as a fraction of the chord
        self.c = self.chord_length   # Chord length
        self.x = np.linspace(0, self.c, self.resolution)  # x coordinates

        # Calculate camber line and thickness
        self.yc = self.camber_line(self.m, self.p, self.c, self.x)
        self.yt = self.thickness(self.t, self.c, self.x)

        # Combine camber and thickness to form the airfoil surfaces
        self.yu = self.yc + self.yt
        self.yl = self.yc - self.yt

        # Rotate the airfoil
        self.x, self.yu = np.dot(self.rotation_matrix, np.array([self.x, self.yu]))
        self.x, self.yl = np.dot(self.rotation_matrix, np.array([self.x, self.yl]))

        # Create a grid
        self.grid_x, self.grid_y = np.mgrid[self.cube_lb[0]:self.cube_rt[0]:self.span, self.cube_lb[1]:self.cube_rt[1]:self.span]

        # Create a mask for points within the airfoil
        self.mask_wing = (self.grid_y >= np.interp(self.grid_x, self.x, self.yl)) & (self.grid_y <= np.interp(self.grid_x, self.x, self.yu)) & (self.grid_x <= (self.c*np.cos(self.attack_angle)))
        self.mask_bound = ~((self.grid_y < (self.cube_rt[1]-self.bound_layer*self.span)) & (self.grid_y > (self.cube_lb[1]+self.bound_layer*self.span))\
              & (self.grid_x > (self.cube_lb[0]+self.bound_layer*self.span)) & (self.grid_x < (self.cube_rt[0]-self.bound_layer*self.span)))
        self.mask_fluid = ~self.mask_wing & ~self.mask_bound & (self.grid_y < (self.cube_rt[1]-(self.bound_layer+self.vacum_layer)*self.span))
        
        # Extract particle locations
        self.wing_particle_x = self.grid_x[self.mask_wing]
        self.wing_particle_y = self.grid_y[self.mask_wing]

        # Shift the airfoil to the given position
        self.x += self.shift_pos[0]
        self.yu += self.shift_pos[1]
        self.yl += self.shift_pos[1]
        self.wing_particle_x += self.shift_pos[0]
        self.wing_particle_y += self.shift_pos[1]

        # Extracr boundary particle locations
        self.bound_particle_x = self.grid_x[self.mask_bound]
        self.bound_particle_y = self.grid_y[self.mask_bound]

        # Extract fluid cube locations
        self.fluid_particle_x = self.grid_x[self.mask_fluid]
        self.fluid_particle_y = self.grid_y[self.mask_fluid]

        # Output data
        self.wing_num = self.wing_particle_x.shape[0]
        self.wing_pos = np.stack((self.wing_particle_x.reshape(-1), self.wing_particle_y.reshape(-1)), -1)
        self.fluid_num = self.fluid_particle_x.shape[0]
        self.fluid_pos = np.stack((self.fluid_particle_x.reshape(-1), self.fluid_particle_y.reshape(-1)), -1)
        self.bound_num = self.bound_particle_x.shape[0]
        self.bound_pos = np.stack((self.bound_particle_x.reshape(-1), self.bound_particle_y.reshape(-1)), -1)

        print('Done!')