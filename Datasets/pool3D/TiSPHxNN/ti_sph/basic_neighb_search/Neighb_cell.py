import taichi as ti
import numpy as np
from ..basic_op.type import *
from ..basic_op.vec_op import *
from ..basic_obj.Obj_Particle import Particle
from ..basic_solvers.Solver import Solver

OUT_OF_RANGE = -1

@ti.data_oriented
class Neighb_cell(Solver):
    def __init__(
            self,
            obj: Particle, # Particle class
    ):
        # get all parameters from obj
        self.obj = obj
        self.dim = ti.static(obj.getObjWorld().g_dim)
        self.cell_size = obj.getObjWorld().support_radius
        self.lb = obj.getObjWorld().g_space_lb
        self.rt = obj.getObjWorld().g_space_rt
        self.part_num = obj.getObjPartNum()
        self.stack_top = obj.getObjStackTop()
        self.pos = obj.pos

        self.attach_to_obj(obj)
        self.parameter_cehck()

        # 计算网格数量(cell_num_vec, cell_num) 并准备好网格编码器(cell_coder)
        self.cell_num = val_i()
        self.cell_num_vec = vecx_i(self.dim[None])
        self.cell_coder = vecx_i(self.dim[None])
        self.calculate_cell_param()
        
        self.part_num_in_cell = ti.field(ti.i32, (self.cell_num[None]))
        self.cell_begin_pointer = ti.field(ti.i32, (self.cell_num[None]))

        self.cell_id_of_part = ti.field(ti.i32, (self.getObj().getObjPartNum()))
        self.part_pointer_shift = ti.field(ti.i32, (self.getObj().getObjPartNum())) # 用于计算part在cell中的偏移
        self.part_id_container = ti.field(ti.i32, (self.getObj().getObjPartNum()))

    
    def attach_to_obj(self, obj):
        self.obj = obj
        # if obj does not have neighb_cell, then create it and attach self to it
        if not hasattr(obj, "neighb_cell"):
            obj.neighb_cell = self
        else:
            raise Exception("obj already has neighb_cell")

    def parameter_cehck(self):
        if self.dim[None] != 2 and self.dim[None] != 3:
            raise Exception("dim must be 2 or 3")
        if self.cell_size[None] <= 0:
            raise Exception("cell_size must be positive")
        if self.lb[None].n != self.rt[None].n:
            raise Exception("lb and rt must have the same shape")
        if self.lb.n != self.dim[None]:
            raise Exception("lb and rt must have the same shape as dim")
        for i in range(self.dim[None]):
            if self.lb[None][i] >= self.rt[None][i]:
                raise Exception("lb must be smaller than rt")

    # 计算网格数量并准备好网格编码器
    @ti.kernel
    def calculate_cell_param(self):
        dim = ti.static(self.cell_coder[None].n)
        assign(self.cell_num_vec[None], ti.ceil(
            (self.rt[None] - self.lb[None]) / self.cell_size[None]).cast(ti.i32))
        for i in ti.static(range(dim)):
            self.cell_coder[None][i] = 1
        self.cell_num[None] = 1
        for i in ti.static(range(dim)):
            self.cell_coder[None][i] = self.cell_num[None]
            self.cell_num[None] *= int(self.cell_num_vec[None][i])
    
    @ti.func
    def encode_into_cell(
        self,
        part_pos: ti.template(),  # 粒子的位置
    ):
        return ti.floor((part_pos - self.lb[None]) // self.cell_size[None]).cast(ti.i32).dot(self.cell_coder[None])

    @ti.func
    def compute_cell_vec(
        self,
        part_pos: ti.template(),  # 粒子的位置
    ):
        return ti.floor((part_pos - self.lb[None]) // self.cell_size[None]).cast(ti.i32)
    
    @ti.func
    def encode_cell_vec(
        self,
        cell_vec: ti.template(),
    ):
        return cell_vec.dot(self.cell_coder[None])
    
    @ti.func
    def within_cell(
        self,
        cell_vec: ti.template(),
    ):
        ans = True
        for i in range(self.dim[None]):
            ans = ans and (0 <= cell_vec[i] < self.cell_num_vec[None][i])   
        return ans
    
    @ti.func
    def getObjPartNum_in_cell(
        self,
        cell_id: ti.i32,
    ):
        return self.part_num_in_cell[cell_id]
    
    @ti.func
    def get_part_id_in_cell(
        self,
        cell_id: ti.template(),
        part_shift: ti.template(),
    ):
        return self.part_id_container[self.cell_begin_pointer[cell_id] + part_shift]

    @ti.kernel
    def update_part_in_cell(
            self,
    ):
        for cell_id in range(self.cell_num[None]):
            self.part_num_in_cell[cell_id] = 0 # 清空 网格粒子数量计数器
        
        for part_id in range(self.obj.tiGetObjStackTop()):
            cell_id = self.encode_into_cell(self.pos[part_id]) # 计算粒子所在网格(编码后)
            if 0 < cell_id < self.cell_num[None]: # 如果粒子在网格范围内则 
                self.cell_id_of_part[part_id] = cell_id # 记录粒子所在网格
                self.part_pointer_shift[part_id] = ti.atomic_add(self.part_num_in_cell[cell_id], 1) # 网格粒子数量计数器 + 1
            else: # 如果粒子不在网格范围内则标记为出界
                self.part_pointer_shift[part_id] = OUT_OF_RANGE
        
        timer = 0
        for cell_id in range(self.cell_num[None]):
            if self.part_num_in_cell[cell_id] > 0: # 如果网格内有粒子则 设置赋予初始指针
                self.cell_begin_pointer[cell_id] = ti.atomic_add(timer, self.part_num_in_cell[cell_id])
        
        for part_id in range(self.obj.tiGetObjStackTop()):
            cell_id = self.cell_id_of_part[part_id]
            if self.part_pointer_shift[part_id] != OUT_OF_RANGE:
                part_pointer = self.cell_begin_pointer[cell_id] + self.part_pointer_shift[part_id]
                self.part_id_container[part_pointer] = part_id
    




    # DEBUG FUNCTION
    @ti.kernel
    def get_part_in_cell(
            self,
    ):
        sum = 0
        for cell_id in range(self.cell_num[None]):
            if self.part_num_in_cell[cell_id] > 0:
                ti.atomic_add(sum, self.part_num_in_cell[cell_id])
                print("There are ", self.part_num_in_cell[cell_id], " particles in cell ", cell_id)
        print("There are ", sum, " particles in total")



    # @ti.func
    # def loop_neighbors(self, part_pos:ti.template(), range:ti.template(), task:ti.template(), ret:ti.template()):
    #     # part_pos: 2/3 dim vector, float
    #     # range: float value
    #     # task: function
    #     # ret: return value, a specific array
    #     cell_id = self.encode_into_cell(part_pos)

