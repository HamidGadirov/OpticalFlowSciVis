import taichi as ti
from typing import List

from .Obj_Particle import Particle
from .Obj import Obj

from ..basic_op.type import *
from ..basic_world import World
from ..basic_data_generator.Cube_generator import Cube_generator
from ..basic_neighb_search.Neighb_search import Neighb_search
from ..basic_solvers.Solver_df import DF_solver

DEFAULT = -1

@ti.data_oriented
class Sense_grid(Particle):

    FIXED_GRID = 0
    FIXED_RES = 1
    def __init__(self, 
                 world: World,  # int
                 cell_size: ti.template(), # float | naming logic: the grid is composed of cells, and each cell has a size
                 neighb_pool_size: ti.template(), # int
                 # specifing the type of sense region
                 type: int = FIXED_GRID, # can also be FIXED_RES
                 # Parameters for FIXED_GRID type 
                 grid_lb: ti.template() = DEFAULT, # vector<dim, float> FIXED_GRID
                 grid_rt: ti.template() = DEFAULT, # vector<dim, float> FIXED_GRID
                 # Parameters for FIXED_RES type
                 grid_res: ti.template() = DEFAULT, # val_i FIXED_RES
                 grid_center: ti.template() = DEFAULT, # vector<dim, float> FIXED_RES
                 ):
        # Reading parameters
        self.setObjWorld(world) 
        
        self.m_neighb_pool_size = neighb_pool_size
        self.m_cell_size = cell_size
        self.m_type = type
        self.m_grid_lb = world.g_space_lb[None] if grid_lb == DEFAULT else grid_lb[None]
        self.m_grid_rt = world.g_space_rt[None] if grid_rt == DEFAULT else grid_rt[None]
        self.m_grid_res = grid_res
        self.m_grid_center = grid_center

        self.m_dim = world.g_dim
        self.m_sensed_parts_list: List[Particle] = []

        # Parameters for father classParticle
        self.m_part_size = self.m_cell_size
        self.smooth_len = val_f(self.m_part_size[None] * 2)

        # Though the data for the grid is stored in a one-dimensional way,
        # we can still use the vector type to represent the shape of the grid.
        # And hopefully, accessing the one-dimensional data with the vector type index.
        self.grid_shape = vecx_i(self.m_dim[None])
        
        if type == Sense_grid.FIXED_GRID:
            # Parameter check
            if grid_center is not DEFAULT or grid_res is not DEFAULT:
                raise Exception("grid_center and grid_res must NOT be given for FIXED_GRID type")
        if type == Sense_grid.FIXED_RES:
            # Parameter check
            if grid_center is DEFAULT or grid_res is DEFAULT:
                raise Exception("grid_center and grid_res must be given for FIXED_RES type")
            temp_grid_size = self.m_grid_res[None] * self.m_cell_size[None]
            temp_grid_lb = self.m_grid_center[None] - temp_grid_size / 2
            temp_grid_rt = self.m_grid_center[None] + temp_grid_size / 2
            self.m_grid_lb = temp_grid_lb
            self.m_grid_rt = temp_grid_rt
            # print("DEBUG grid_lb", self.grid_lb)
            # print("DEBUG grid_rt", self.grid_rt)

        self.generator = Cube_generator(self, self.m_grid_lb, self.m_grid_rt)
        self.node_num = val_i(self.generator.pushed_num_preview(span=self.get_cell_size()[None]))
        
        super().__init__(part_num=self.get_node_num()[None], part_size=self.getObjPartSize(), is_dynamic=False)
        self.setObjWorld(world)

        self.add_array("pos", vecxf(world.getWorldDim()).field())
        self.add_array("size", ti.field(ti.f32))
        self.add_array("clampped", ti.field(ti.f32))
        self.add_array("clampped_rgb", vec3f.field())
        # index is for logging the sequence of particles
        self.add_array("node_index", ti.field(ti.i32), bundle=self.m_dim[None])

        self.add_attr("density_upper_bound", ti.field(ti.f32, ()))
        self.add_attr("density_lower_bound", ti.field(ti.f32, ()))

        sph = ti.types.struct(
        h=ti.f32,
        sig=ti.f32,
        sig_inv_h=ti.f32,
        density=ti.f32,
        )
        self.add_struct("sph", sph)

        self.generator.push_pos()
        self.generator._get_index(self.node_index)
        self.grid_shape.from_numpy(self.generator.get_shape())
        self.set_val(to_arr=self.size, num=self.get_node_num()[None], val=self.m_part_size)
        self.update_stack_top(self.get_node_num()[None])

        self.neighb_search = Neighb_search(self, neighb_pool_size)
        self.df_solver = DF_solver(self)
        
    def add_sensed_particles(self, particles: Particle):
        if particles not in self.m_sensed_parts_list:
            self.m_sensed_parts_list.append(particles)
        else:
            raise Exception("particle already added")
        self.neighb_search.add_neighb_obj(particles, self.smooth_len)
        self.neighb_search.search_neighbors()

    def get_cell_size(self):
        return self.getObjPartSize()
    
    def get_node_num(self):
        return self.node_num
    
    @ti.func
    def ti_get_index(self, id):
        return self.node_index[id]
    
    @ti.func
    def ti_get_node_num(self):
        return self.node_num
    
    def step(self):
        self.neighb_search.search_neighbors()
        self.clear(self.sph.density)
        for sensed_parts in self.m_sensed_parts_list:
            self.df_solver.loop_neighb(self.neighb_search.neighb_pool, sensed_parts, self.df_solver.inloop_accumulate_density)
        self.density_lower_bound[None] = 0
        self.density_upper_bound[None] = 1000
        self.clamp_val(self.sph.density, self.density_lower_bound, self.density_upper_bound)
    
    @ti.kernel
    def clamp_val(self, arr: ti.template(), lower: ti.template(), upper: ti.template()):
        for i in range(self.ti_get_node_num()[None]):
            self.clampped[i] = ti.min(ti.max(arr[i] / (upper[None] - lower[None]),0),1)
            self.clampped_rgb[i] = [self.clampped[i],self.clampped[i],self.clampped[i]]
