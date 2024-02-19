import taichi as ti

from .modules import Mod_AttrAllo
from .modules import Mod_DataOp
from .modules import Mod_GetAndSet
from .modules import Mod_NeighbSearch
from .modules import Mod_Solvers

from ..Obj import Obj
from ...basic_op.type import *

@ti.data_oriented
class Particle(Obj, Mod_Solvers, Mod_DataOp, Mod_AttrAllo, Mod_NeighbSearch, Mod_GetAndSet):
    def __init__(
        self,
        part_num: int,
        part_size: ti.template(),
        is_dynamic: bool = True,
    ):

        self.m_world = None
        self.m_id = None    
        
        Obj.__init__(self, is_dynamic)
        Mod_Solvers.__init__(self)

        if (part_num <= 0):
            raise ValueError("part_num must be larger than 0")

        # data structure management
        self.m_part_num = val_i(part_num)
        self.m_stack_top = val_i(0)
        self.m_if_stack_open = False
        self.m_stack_open_num = val_i(0)
        # shared attributes
        self.m_part_size = part_size

        # modules
        self.m_neighb_search = None
        
        # data structure
        self.m_attr_list = {}
        self.m_array_list = {}
        self.m_struct_list = {}

        # TODO
        self.m_delete_list = ti.field(ti.i32, self.m_part_num[None]+1)

    

    # TODO
    def delete_outbounded_particles(self):
        self.clear(self.m_delete_list)
        self.log_tobe_deleted_particles()
    # TODO
    @ti.kernel
    def log_tobe_deleted_particles(self):
        counter = ti.static(self.m_delete_list[self.m_delete_list.shape[0]])
        for part_id in range(self.tiGetObjStackTop()):
            if self.has_negative(self.pos[part_id]-self.m_world.space_lb[None]) or self.has_positive(self.pos[part_id]-self.m_world.space_rt[None]):
                self.m_delete_list[ti.atomic_add(self.m_delete_list[counter],1)] = part_id
    # TODO
    def move(original: ti.i32, to: ti.i32):
        pass

    @ti.kernel
    def color_code_part(self: ti.template(), arr: ti.template(), lower: ti.f32, upper: ti.f32):
        for i in range(self.tiGetObjStackTop()):
            val = ti.min(ti.max(ti.math.length(arr[i]) / (upper - lower), 0), 1)
            self.rgb[i][0] = val
            self.rgb[i][1] = val