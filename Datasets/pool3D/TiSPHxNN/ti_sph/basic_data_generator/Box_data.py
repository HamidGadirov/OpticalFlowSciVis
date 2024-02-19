import taichi as ti
import numpy as np

from .Data_generator import Data_generator

@ti.data_oriented
class Box_data(Data_generator):
    def __init__(self, lb: ti.Vector, rt: ti.Vector, span: ti.f32, layers: ti.i32):
        self.lb = lb
        self.rt = rt
        self.span = span
        self.layers = layers
        self.dim = ti.static(lb.n)

        self.num = self.pushed_num_preview(span)
        pos_data = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.num)
        self.generate_pos(pos_data)
        self.pos = pos_data

    @ti.kernel
    def pushed_num_preview(self, span: ti.f32) -> ti.i32:
        current_node_num = 0
        pushed_node_seq_coder = ti.Vector([0, 0, 0])

        pushed_node_seq = int(ti.ceil((self.rt - self.lb) / span))
        pushed_node_seq_offset = int(ti.ceil((self.rt - self.lb) / span)) + (self.layers * 2)

        for i in ti.static(range(self.dim)):
            if pushed_node_seq[i] == 0:
                pushed_node_seq[i] = 1  # at least push one

        tmp = 1
        for i in ti.static(range(self.dim)):
            pushed_node_seq_coder[i] = tmp
            tmp *= pushed_node_seq_offset[i]

        pushed_node_num = 1
        pushed_node_num_solid = 1
        for i in ti.static(range(self.dim)):
            pushed_node_num *= pushed_node_seq[i]
            pushed_node_num_solid *= pushed_node_seq_offset[i]
        pushed_node_num = pushed_node_num_solid - pushed_node_num
        new_node_num = current_node_num + pushed_node_num

        inc = ti.Vector([current_node_num])
        for i in range(pushed_node_num_solid):
            tmp = i
            a = self.rt - self.lb
            flag = False
            dim_check = 0
            for j in ti.static(range(self.dim - 1, -1, -1)):
                a[j] = tmp // pushed_node_seq_coder[j]
                tmp = tmp % pushed_node_seq_coder[j]
            t1 = (a - 1) - pushed_node_seq
            t2 = a - self.layers
            
            for i in ti.static(range(t1.n)):
                if t1[i] > 0:
                    flag = True
            for i in ti.static(range(t2.n)):
                if t2[i] < 0:
                    flag = True
            if flag:
                index = ti.atomic_add(inc[0], 1)
        return inc[0]

    @ti.kernel
    def generate_pos(self, _data: ti.template()):
        current_node_num = 0
        pushed_node_seq_coder = ti.Vector([0, 0, 0])

        pushed_node_seq = int(ti.ceil((self.rt - self.lb) / self.span))
        pushed_node_seq_offset = int(ti.ceil((self.rt - self.lb) / self.span)) + (self.layers * 2)

        for i in ti.static(range(self.dim)):
            if pushed_node_seq[i] == 0:
                pushed_node_seq[i] = 1  # at least push one

        tmp = 1
        for i in ti.static(range(self.dim)):
            pushed_node_seq_coder[i] = tmp
            tmp *= pushed_node_seq_offset[i]

        pushed_node_num = 1
        pushed_node_num_solid = 1
        for i in ti.static(range(self.dim)):
            pushed_node_num *= pushed_node_seq[i]
            pushed_node_num_solid *= pushed_node_seq_offset[i]
        pushed_node_num = pushed_node_num_solid - pushed_node_num
        new_node_num = current_node_num + pushed_node_num

        inc = ti.Vector([current_node_num])
        for i in range(pushed_node_num_solid):
            tmp = i
            a = self.rt - self.lb
            flag = False
            dim_check = 0
            for j in ti.static(range(self.dim - 1, -1, -1)):
                a[j] = tmp // pushed_node_seq_coder[j]
                tmp = tmp % pushed_node_seq_coder[j]
            t1 = (a - 1) - pushed_node_seq
            t2 = a - self.layers
            
            for i in ti.static(range(t1.n)):
                if t1[i] > 0:
                    flag = True
            for i in ti.static(range(t2.n)):
                if t2[i] < 0:
                    flag = True
            if flag:
                index = ti.atomic_add(inc[0], 1)
                for j in ti.static(range(self.dim - 1, -1, -1)):
                    _data[index][j] = (a[j] - self.layers) * self.span + self.lb[j]