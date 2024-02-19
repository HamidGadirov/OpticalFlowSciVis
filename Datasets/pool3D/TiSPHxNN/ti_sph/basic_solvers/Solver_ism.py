import taichi as ti
import math
from .sph_funcs import *
from .Solver_sph import SPH_solver
from .Solver_multiphase import Multiphase_solver
from ..basic_op.type import *
from ..basic_obj.Obj_Particle import Particle

from typing import List

GREEN = ti.Vector([0.0, 1.0, 0.0])
WHITE = ti.Vector([1.0, 1.0, 1.0])
DARK = ti.Vector([0.0, 0.0, 0.0])

@ti.data_oriented
class Implicit_mixture_solver(Multiphase_solver):
    def __init__(self, obj: Particle, Cd: ti.f32, Cf: ti.f32, k_vis_inter: ti.f32, k_vis_inner: ti.f32, world):
        
        super().__init__(obj, Cf, world)

        self.Cd = val_f(Cd)
        self.k_vis_inter = val_f(k_vis_inter)
        self.k_vis_inner = val_f(k_vis_inner)

    @ti.kernel
    def ditribute_acc_pressure_2_phase(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.acc[part_id, phase_id] += self.obj.mixture.acc_pressure[part_id] * \
                    (self.Cd[None] + ((1 - self.Cd[None]) * (self.obj.rest_density[part_id]/self.world.g_phase_rest_density[None][phase_id])))

    @ti.func
    def inloop_add_phase_acc_vis(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        x_ij = self.obj.pos[part_id] - neighb_obj.pos[neighb_part_id]
        if bigger_than_zero(cached_dist):
            v_ij = self.obj.vel[part_id] - neighb_obj.vel[neighb_part_id]
            for phase_id in range(self.phase_num[None]):
                v_ki_mj = self.obj.phase.vel[part_id, phase_id] - neighb_obj.vel[neighb_part_id]
                self.obj.phase.acc[part_id, phase_id] += 2*(2+self.obj.m_world.g_dim[None]) * neighb_obj.volume[neighb_part_id] * \
                    ((self.k_vis_inner[None] * (1-self.Cd[None]) *  v_ki_mj) + (self.k_vis_inter[None] * self.Cd[None] * v_ij)).dot(x_ij) * cached_grad_W \
                    / (cached_dist**2) 

    # def update_phase_change(self):
    #     self.update_phase_change_ker()
        # self.clear_phase_acc()
        # self.loop_neighb(self.obj.m_neighb_search.neighb_pool, self.obj, self.inloop_update_phase_change_from_all)
        # self.re_arrange_phase_vel()
    
    # @ti.func
    # def inloop_update_phase_change_from_all(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
    #     cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
    #     cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
    #     if bigger_than_zero(cached_dist) and (self.obj.mixture[part_id].flag_negative_val_frac == 0 and neighb_obj.mixture[neighb_part_id].flag_negative_val_frac == 0):
    #         for phase_id in range(self.phase_num[None]):
    #             val_frac_ij = self.obj.phase.val_frac[part_id, phase_id] - neighb_obj.phase.val_frac[neighb_part_id, phase_id]
    #             x_ij = self.obj.pos[part_id] - neighb_obj.pos[neighb_part_id]
    #             diffuse_val_change = self.dt[None] * self.Cf * val_frac_ij * neighb_obj.volume[neighb_part_id] * cached_grad_W.dot(x_ij) / (cached_dist**2)
    #             drift_val_change = -self.dt[None] * neighb_obj.volume[neighb_part_id] * \
    #                 (self.obj.phase.val_frac[part_id, phase_id] * self.obj.phase.drift_vel[part_id, phase_id] + \
    #                 neighb_obj.phase.val_frac[neighb_part_id, phase_id] * neighb_obj.phase.drift_vel[neighb_part_id, phase_id]).dot(cached_grad_W)
    #             val_frac_change = diffuse_val_change + drift_val_change
    #             if val_frac_change > 0:
    #                 self.obj.phase.acc[part_id, phase_id] += val_frac_change * neighb_obj.phase.vel[neighb_part_id, phase_id]

    # @ti.kernel
    # def re_arrange_phase_vel(self):
    #     for part_id in range(self.obj.tiGetObjStackTop()):
    #         if self.obj.mixture[part_id].flag_negative_val_frac == 0:
    #             for phase_id in range(self.phase_num[None]):
    #                 val_frac_remain = self.obj.phase.val_frac[part_id, phase_id] + self.obj.phase.val_frac_out[part_id, phase_id]
    #                 self.obj.phase.val_frac[part_id, phase_id] = val_frac_remain + self.obj.phase.val_frac_in[part_id, phase_id]
    #                 if bigger_than_zero(self.obj.phase.val_frac[part_id, phase_id]):
    #                     self.obj.phase.vel[part_id, phase_id] = \
    #                         self.obj.phase.acc[part_id, phase_id] + (val_frac_remain*self.obj.phase.vel[part_id, phase_id]) / self.obj.phase.val_frac[part_id, phase_id]
                    # else:
                    #     self.obj.phase.vel[part_id, phase_id] = self.obj.phase.acc[part_id, phase_id]/self.obj.phase.val_frac[part_id, phase_id]





