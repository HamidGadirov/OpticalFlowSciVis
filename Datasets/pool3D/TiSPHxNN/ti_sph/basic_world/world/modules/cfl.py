import taichi as ti
import taichi.math as tm

from ....basic_solvers import sph_funcs

def init_cfl(self):
    self.cfl_list = []
    for part_obj in self.part_obj_list:
        if part_obj.m_is_dynamic is not False:
            self.cfl_list.append(part_obj)

@ti.kernel
def find_max_vec(self: ti.template(), data: ti.template(), loop_range: ti.i32)->ti.f32:
    tmp_val = 0.0
    for i in range(loop_range):
        ti.atomic_max(tmp_val, tm.length(data[i]))
    return tmp_val

def cfl_dt(self, cfl_factor: float, max_dt: float):
    max_vel = sph_funcs.INF_SMALL
    for part_obj in self.cfl_list:
        max_vel = max(self.find_max_vec(part_obj.vel, part_obj.getObjStackTop()), max_vel)
    new_dt = min(max_dt, self.g_part_size[None] / max_vel * cfl_factor)
    self.setWorldDt(new_dt)
    return new_dt, max_vel

def get_cfl_dt_obj(self, part_obj, cfl_factor: float, max_dt: float):
    max_vel = sph_funcs.INF_SMALL
    max_vel = max(self.find_max_vec(part_obj.vel, part_obj.getObjStackTop()), max_vel)
    new_dt = min(max_dt, self.g_part_size[None] / max_vel * cfl_factor)
    return new_dt, max_vel