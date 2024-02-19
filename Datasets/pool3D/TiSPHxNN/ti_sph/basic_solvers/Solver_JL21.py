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
class JL21_mixture_solver(Multiphase_solver):
    def __init__(self, obj: Particle, kd: ti.f32, Cf: ti.f32, k_vis: ti.f32, world):
        
        super().__init__(obj, Cf, world)

        self.kd = val_f(kd)
        self.k_vis = val_f(k_vis)

    @ti.kernel
    def clear_vis_force(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            self.obj.sph.viscosity_force[part_id] *= 0
    
    @ti.kernel
    def clear_pressure_force(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            self.obj.sph.pressure_force[part_id] *= 0

    @ti.func
    def inloop_add_force_pressure(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            part_volume = self.obj.mass[part_id] / self.obj.sph[part_id].density
            neighb_part_volume = neighb_obj.mass[neighb_part_id] / neighb_obj.sph[neighb_part_id].density
            self.obj.sph.pressure_force[part_id] -= part_volume * neighb_part_volume * (self.obj.pressure[part_id] + neighb_obj.pressure[neighb_part_id]) * cached_grad_W
    
    @ti.func
    def inloop_add_force_vis(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            A_ij = self.obj.vel[part_id] - neighb_obj.vel[neighb_part_id]
            x_ij = self.obj.pos[part_id] - neighb_obj.pos[neighb_part_id]
            PI_ij = - (self.k_vis[None]) * \
                ( ti.min(0, A_ij.dot(x_ij)) / ((cached_dist**2)+INF_SMALL*(self.obj.sph[part_id].h**2)) )
            self.obj.sph.viscosity_force[part_id] += - self.obj.mass[part_id] * neighb_obj.mass[neighb_part_id] * PI_ij * cached_grad_W

    @ti.kernel
    def update_phase_vel(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            for phase_id in range(self.phase_num[None]):
                phase_mass = self.world.g_phase_rest_density[None][phase_id] * self.obj.volume[part_id]
                self.obj.phase.vel[part_id, phase_id] += self.dt[None] * (self.obj.acc[part_id] +\
                    (self.obj.sph.viscosity_force[part_id]+self.obj.sph.pressure_force[part_id])/phase_mass)

        for part_id in range(self.obj.tiGetObjStackTop()):
            self.obj.vel[part_id] *= 0
            for phase_id in range(self.phase_num[None]):
                self.obj.vel[part_id] += self.obj.phase.vel[part_id, phase_id] * self.obj.phase.val_frac[part_id, phase_id]

        for part_id in range(self.obj.tiGetObjStackTop()):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.vel[part_id, phase_id] += - self.kd[None] * self.dt[None] * \
                    (self.obj.rest_density[part_id]/self.world.g_phase_rest_density[None][phase_id]) *\
                    (self.obj.phase.vel[part_id, phase_id] - self.obj.vel[part_id])
        
        for part_id in range(self.obj.tiGetObjStackTop()):
            self.obj.vel[part_id] *= 0
            for phase_id in range(self.phase_num[None]):
                self.obj.vel[part_id] += self.obj.phase.vel[part_id, phase_id] * self.obj.phase.val_frac[part_id, phase_id]
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.drift_vel[part_id, phase_id] = self.obj.phase.vel[part_id, phase_id] - self.obj.vel[part_id]

    @ti.kernel
    def vis_force_2_acc(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            self.obj.acc[part_id] += self.obj.sph.viscosity_force[part_id] / self.obj.mass[part_id]
    
    @ti.kernel
    def pressure_force_2_acc(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            self.obj.acc[part_id] += self.obj.sph.pressure_force[part_id] / self.obj.mass[part_id]

    @ti.kernel
    def acc_2_vel(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            self.obj.vel[part_id] += self.dt[None] * self.obj.acc[part_id]

    # @ti.kernel
    # def ditribute_acc_pressure_2_phase(self):
    #     for part_id in range(self.obj.tiGetObjStackTop()):
    #         for phase_id in range(self.phase_num[None]):
    #             self.obj.phase.acc[part_id, phase_id] += self.obj.mixture.acc_pressure[part_id] * \
    #                 (self.Cd + ((1 - self.Cd) * (self.obj.rest_density[part_id]/self.world.g_phase_rest_density[None][phase_id])))

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





