import taichi as ti
import math
from .sph_funcs import *
from .Solver_sph import SPH_solver
from ..basic_op.type import *
from ..basic_obj.Obj_Particle import Particle
from typing import List

GREEN = ti.Vector([0.0, 1.0, 0.0])
WHITE = ti.Vector([1.0, 1.0, 1.0])
DARK = ti.Vector([0.0, 0.0, 0.0])

@ti.data_oriented
class Multiphase_solver(SPH_solver):
    def __init__(self, obj: Particle, Cf: ti.f32, world):
        
        super().__init__(obj)

        self.phase_num = world.g_phase_num
        self.world = world
        self.Cf = Cf

    @ti.kernel
    def clear_phase_acc(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.acc[part_id, phase_id] *= 0
                # self.obj.phase.val_frac_tmp[part_id, phase_id] = self.obj.phase.val_frac[part_id, phase_id]
    
    @ti.kernel
    def add_phase_acc_gravity(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.acc[part_id, phase_id] += self.world.g_gravity[None]
                
    @ti.kernel
    def phase_acc_2_phase_vel(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.vel[part_id, phase_id] += self.obj.phase.acc[part_id, phase_id] * self.world.g_dt[None]

    @ti.func
    def inloop_update_phase_change(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist) and (self.obj.mixture[part_id].flag_negative_val_frac == 0 and neighb_obj.mixture[neighb_part_id].flag_negative_val_frac == 0):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.val_frac_tmp[part_id, phase_id] -= self.dt[None] * neighb_obj.volume[neighb_part_id] * \
                (self.obj.phase.val_frac[part_id, phase_id] * self.obj.phase.drift_vel[part_id, phase_id] + \
                 neighb_obj.phase.val_frac[neighb_part_id, phase_id] * neighb_obj.phase.drift_vel[neighb_part_id, phase_id]).dot(cached_grad_W)
    
    @ti.kernel
    def release_negative(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            self.obj.mixture[part_id].flag_negative_val_frac = 0
    

    ''' ######################## PHASE UPDATE (SCHEME 1) ######################## '''

    def update_val_frac(self):
        self.clear_val_frac_tmp()
        self.loop_neighb(self.obj.m_neighb_search.neighb_pool, self.obj, self.inloop_update_phase_change_from_drift)
        self.loop_neighb(self.obj.m_neighb_search.neighb_pool, self.obj, self.inloop_update_phase_change_from_diffuse)
        while self.check_negative() == 0:
            self.clear_val_frac_tmp()
            self.loop_neighb(self.obj.m_neighb_search.neighb_pool, self.obj, self.inloop_update_phase_change_from_drift)
            self.loop_neighb(self.obj.m_neighb_search.neighb_pool, self.obj, self.inloop_update_phase_change_from_diffuse)
        self.update_phase_change()
        self.release_unused_drift_vel()
        self.release_negative()
        self.regularize_val_frac()
        self.update_rest_density_and_mass()

    @ti.kernel
    def clear_val_frac_tmp(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.val_frac_in[part_id, phase_id] = 0
                self.obj.phase.val_frac_out[part_id, phase_id] = 0

    @ti.func
    def inloop_update_phase_change_from_drift(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist) and (self.obj.mixture[part_id].flag_negative_val_frac == 0 and neighb_obj.mixture[neighb_part_id].flag_negative_val_frac == 0):
            for phase_id in range(self.phase_num[None]):
                val_frac_change = -self.dt[None] * neighb_obj.volume[neighb_part_id] * \
                (self.obj.phase.val_frac[part_id, phase_id] * self.obj.phase.drift_vel[part_id, phase_id] + \
                 neighb_obj.phase.val_frac[neighb_part_id, phase_id] * neighb_obj.phase.drift_vel[neighb_part_id, phase_id]).dot(cached_grad_W)
                if val_frac_change < 0: # out
                    self.obj.phase.val_frac_out[part_id, phase_id] += val_frac_change
                else: # in
                    self.obj.phase.val_frac_in[part_id, phase_id] += val_frac_change

    @ti.func
    def inloop_update_phase_change_from_diffuse(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist) and (self.obj.mixture[part_id].flag_negative_val_frac == 0 and neighb_obj.mixture[neighb_part_id].flag_negative_val_frac == 0):
            for phase_id in range(self.phase_num[None]):
                val_frac_ij = self.obj.phase.val_frac[part_id, phase_id] - neighb_obj.phase.val_frac[neighb_part_id, phase_id]
                x_ij = self.obj.pos[part_id] - neighb_obj.pos[neighb_part_id]
                val_frac_change = self.dt[None] * self.Cf * val_frac_ij * neighb_obj.volume[neighb_part_id] * cached_grad_W.dot(x_ij) / (cached_dist**2)
                if val_frac_change < 0: # out
                    self.obj.phase.val_frac_out[part_id, phase_id] += val_frac_change
                else: # in
                    self.obj.phase.val_frac_in[part_id, phase_id] += val_frac_change

    @ti.kernel
    def check_negative(self) -> ti.i32:
        all_positive = 1
        for part_id in range(self.obj.tiGetObjStackTop()):
            if self.obj.mixture[part_id].flag_negative_val_frac == 0:
                for phase_id in range(self.phase_num[None]):
                    if self.obj.phase.val_frac[part_id, phase_id] + self.obj.phase.val_frac_out[part_id, phase_id] + self.obj.phase.val_frac_in[part_id, phase_id] < 0:
                        self.obj.mixture[part_id].flag_negative_val_frac = 1
                        all_positive = 0
        return all_positive

    @ti.kernel
    def release_negative(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            self.obj.mixture[part_id].flag_negative_val_frac = 0

    @ti.kernel
    def release_unused_drift_vel(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            if not self.obj.mixture[part_id].flag_negative_val_frac == 0:
                for phase_id in range(self.phase_num[None]):
                    self.obj.phase.vel[part_id, phase_id] = self.obj.vel[part_id]

    @ti.kernel
    def update_vel_from_phase_vel(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            self.obj.vel[part_id] *= 0
            for phase_id in range(self.phase_num[None]):
                # if bigger_than_zero(self.obj.phase.val_frac[part_id, phase_id]):
                self.obj.vel[part_id] += self.obj.phase.vel[part_id, phase_id] * self.obj.phase.val_frac[part_id, phase_id]
            for phase_id in range(self.phase_num[None]):
                # if not bigger_than_zero(self.obj.phase.val_frac[part_id, phase_id]):
                #     self.obj.phase.vel[part_id, phase_id] = self.obj.vel[part_id]
                self.obj.phase.drift_vel[part_id, phase_id] = self.obj.phase.vel[part_id, phase_id] - self.obj.vel[part_id]

    @ti.kernel
    def regularize_val_frac(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            frac_sum = 0.0
            for phase_id in range(self.phase_num[None]):
                frac_sum += self.obj.phase.val_frac[part_id, phase_id]
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.val_frac[part_id, phase_id] /= frac_sum

    @ti.kernel
    def update_rest_density_and_mass(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            rest_density = 0
            for phase_id in range(self.phase_num[None]):
                rest_density += self.obj.phase.val_frac[part_id, phase_id] * self.world.g_phase_rest_density[None][phase_id]
            self.obj.rest_density[part_id] = rest_density
            self.obj.mass[part_id] = self.obj.rest_density[part_id] * self.obj.volume[part_id]

    @ti.kernel
    def update_phase_change(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.val_frac[part_id, phase_id] += (self.obj.phase.val_frac_in[part_id, phase_id] + self.obj.phase.val_frac_out[part_id, phase_id])

    @ti.kernel
    def update_color(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            color = ti.Vector([0.0, 0.0, 0.0])
            for phase_id in range(self.phase_num[None]):
                for rgb_id in ti.static(range(3)):
                    color[rgb_id] += self.obj.phase.val_frac[part_id, phase_id] * self.world.g_phase_color[phase_id][rgb_id]
            for rgb_id in range(self.phase_num[None]):
                color[rgb_id] = ti.min(color[rgb_id], 1.0)
            self.obj.rgb[part_id] = color

    ''' ######################## PHASE UPDATE (SCHEME 2) ######################## '''

    def update_val_frac_lamb(self):
        self.reset_lambda()
        self.clear_val_frac_tmp()
        self.loop_neighb(self.obj.m_neighb_search.neighb_pool, self.obj, self.inloop_update_phase_change_from_drift_lamb)
        self.loop_neighb(self.obj.m_neighb_search.neighb_pool, self.obj, self.inloop_update_phase_change_from_diffuse_lamb)
        loop=0
        while self.check_negative_lamb() == 0:
            loop+=1
            # print('triggered!', loop)
            self.clear_val_frac_tmp()
            self.loop_neighb(self.obj.m_neighb_search.neighb_pool, self.obj, self.inloop_update_phase_change_from_drift_lamb)
            self.loop_neighb(self.obj.m_neighb_search.neighb_pool, self.obj, self.inloop_update_phase_change_from_diffuse_lamb)
        self.update_phase_change()
        self.regularize_val_frac_lamb()
        self.update_rest_density_and_mass()

    @ti.kernel
    def reset_lambda(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
                self.obj.mixture.lamb[part_id] = 1.0

    @ti.func
    def inloop_update_phase_change_from_drift_lamb(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            lamb_ij = ti.min(self.obj.mixture.lamb[part_id], neighb_obj.mixture.lamb[neighb_part_id])
            for phase_id in range(self.phase_num[None]):
                val_frac_change = -self.dt[None] * neighb_obj.volume[neighb_part_id] * \
                (self.obj.phase.val_frac[part_id, phase_id] * self.obj.phase.drift_vel[part_id, phase_id] + \
                 neighb_obj.phase.val_frac[neighb_part_id, phase_id] * neighb_obj.phase.drift_vel[neighb_part_id, phase_id]).dot(cached_grad_W)
                if val_frac_change < 0: # out
                    self.obj.phase.val_frac_out[part_id, phase_id] += val_frac_change * lamb_ij
                else: # in
                    self.obj.phase.val_frac_in[part_id, phase_id] += val_frac_change * lamb_ij

    @ti.func
    def inloop_update_phase_change_from_diffuse_lamb(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            lamb_ij = ti.min(self.obj.mixture.lamb[part_id], neighb_obj.mixture.lamb[neighb_part_id])
            for phase_id in range(self.phase_num[None]):
                val_frac_ij = self.obj.phase.val_frac[part_id, phase_id] - neighb_obj.phase.val_frac[neighb_part_id, phase_id]
                x_ij = self.obj.pos[part_id] - neighb_obj.pos[neighb_part_id]
                val_frac_change = self.dt[None] * self.Cf * val_frac_ij * neighb_obj.volume[neighb_part_id] * cached_grad_W.dot(x_ij) / (cached_dist**2)
                if val_frac_change < 0: # out
                    self.obj.phase.val_frac_out[part_id, phase_id] += val_frac_change * lamb_ij
                else: # in
                    self.obj.phase.val_frac_in[part_id, phase_id] += val_frac_change * lamb_ij

    @ti.kernel
    def regularize_val_frac_lamb(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            frac_sum = 0.0
            for phase_id in range(self.phase_num[None]):
                if self.obj.phase.val_frac[part_id, phase_id] < 0:
                    # print('negative phase', part_id, phase_id, self.obj.phase.val_frac[part_id, phase_id])
                    self.obj.phase.val_frac[part_id, phase_id] = 0.0
            for phase_id in range(self.phase_num[None]):
                frac_sum += self.obj.phase.val_frac[part_id, phase_id]
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.val_frac[part_id, phase_id] /= frac_sum

    @ti.kernel
    def check_negative_lamb(self) -> ti.i32:
        all_positive = 1
        
        for part_id in range(self.obj.tiGetObjStackTop()):
            this_negative = 0

            for phase_id in range(self.phase_num[None]):
                estimated_val_frac = self.obj.phase.val_frac[part_id, phase_id] + self.obj.phase.val_frac_out[part_id, phase_id] + self.obj.phase.val_frac_in[part_id, phase_id]
                if estimated_val_frac < -INF_SMALL:
                    # print('how negative', self.obj.phase.val_frac[part_id, phase_id] + self.obj.phase.val_frac_out[part_id, phase_id] + self.obj.phase.val_frac_in[part_id, phase_id])
                    this_negative = 1
                    all_positive = 0
            
            if this_negative == 1:
                min_factor = 1.0
                for phase_id in range(self.phase_num[None]):
                    if self.obj.phase.val_frac[part_id, phase_id] + self.obj.phase.val_frac_out[part_id, phase_id] + self.obj.phase.val_frac_in[part_id, phase_id] < 0:
                        ti.atomic_min(min_factor, self.obj.phase.val_frac[part_id, phase_id]/(-(self.obj.phase.val_frac_out[part_id, phase_id] + self.obj.phase.val_frac_in[part_id, phase_id])))
                        # print('min_factor', min_factor)
                self.obj.mixture.lamb[part_id] *= ti.abs(min_factor)
                # print('lamb', self.obj.mixture.lamb[part_id])
        return all_positive

    ''' ######################## STATISTICS ######################## '''

    @ti.kernel
    def statistics_linear_momentum_and_kinetic_energy(self):
        self.obj.statistics_linear_momentum[None] *= 0
        self.obj.statistics_kinetic_energy[None] *= 0
        for part_id in range(self.obj.tiGetObjStackTop()):
            for phase_id in range(self.phase_num[None]):
                self.obj.statistics_linear_momentum[None] += self.obj.phase.vel[part_id, phase_id] * self.obj.phase.val_frac[part_id, phase_id] * self.obj.volume[part_id] * self.world.g_phase_rest_density[None][phase_id]
                self.obj.statistics_kinetic_energy[None] += 0.5 * self.obj.phase.vel[part_id, phase_id].dot(self.obj.phase.vel[part_id, phase_id]) * self.obj.phase.val_frac[part_id, phase_id] * self.obj.volume[part_id] * self.world.g_phase_rest_density[None][phase_id]
        print('statistics linear momentum:', self.obj.statistics_linear_momentum[None])
        print('statistics kinetic energy:', self.obj.statistics_kinetic_energy[None])
    
    def statistics_angular_momentum(self):
        self.clear_phase_acc()
        self.loop_neighb(self.obj.m_neighb_search.neighb_pool, self.obj, self.inloop_compute_angular_momentum)
        self.compute_angular_momentum()
    
    @ti.func
    def inloop_compute_angular_momentum(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.acc[part_id, phase_id] += neighb_obj.volume[neighb_part_id] / 2 * \
                    (neighb_obj.phase.vel[neighb_part_id, phase_id] - self.obj.phase.vel[part_id, phase_id]).cross(cached_grad_W)
    
    @ti.kernel
    def compute_angular_momentum(self):
        self.obj.statistics_angular_momentum[None] *= 0
        for part_id in range(self.obj.tiGetObjStackTop()):
            # vel = ti.Vector([self.obj.vel[part_id][0], self.obj.vel[part_id][1], 0.0])
            # pos = ti.Vector([self.obj.pos[part_id][0], self.obj.pos[part_id][1], 0.0])
            # self.obj.statistics_angular_momentum[None] += vel.cross(pos)*self.obj.mass[part_id]
            for phase_id in range(self.phase_num[None]):
                vel = ti.Vector([self.obj.phase.vel[part_id, phase_id][0], self.obj.phase.vel[part_id, phase_id][1], 0.0])
                pos = ti.Vector([self.obj.pos[part_id][0], self.obj.pos[part_id][1], 0.0])
                self.obj.statistics_angular_momentum[None] += vel.cross(pos) * \
                    self.obj.volume[part_id] * self.world.g_phase_rest_density[None][phase_id] * self.obj.phase.val_frac[part_id, phase_id]
        print('statistics angular momentum:', self.obj.statistics_angular_momentum[None])

    ''' ######################## UTIL ######################## '''

    @ti.kernel
    def cfl_dt(self, cfl_factor: ti.f32, max_dt: ti.f32) -> ti.f32:
        max_vel = 0.0
        for part_id in range(self.obj.tiGetObjStackTop()):
            for phase_id in range(self.phase_num[None]):
                ti.atomic_max(max_vel, ti.math.length(self.obj.phase.vel[part_id, phase_id]))
        new_dt = ti.min(max_dt, self.world.g_part_size[None] / max_vel * cfl_factor)
        return new_dt

    @ti.kernel
    def recover_phase_vel_from_mixture(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.vel[part_id, phase_id] = self.obj.vel[part_id]

    @ti.kernel
    def debug_zero_out_drift_vel(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            for phase_id in range(self.phase_num[None]):
                self.obj.phase.drift_vel[part_id, phase_id] *= 0
                self.obj.phase.vel[part_id, phase_id] = self.obj.vel[part_id]

    @ti.kernel
    def debug_zero_out_small_drift(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            for phase_id in range(self.phase_num[None]):
                if self.obj.phase.val_frac[part_id, phase_id] < INF_SMALL:
                    self.obj.phase.vel[part_id, phase_id] = self.obj.vel[part_id]

    @ti.kernel
    def debug_draw_drift_vel(self, phase:ti.i32):
        max_vel = 0.0
        phase_id = phase
        for part_id in range(self.obj.tiGetObjStackTop()):
            length = ti.math.length(self.obj.phase.drift_vel[part_id, phase_id])/5
            self.obj.rgb[part_id] = ti.Vector([length, length, length])
            # ti.atomic_max(max_vel, ti.math.length(self.obj.phase.drift_vel[part_id, phase_id]))

    @ti.kernel
    def debug_draw_empty_phase(self):
        fact = 0.99999
        for part_id in range(self.obj.tiGetObjStackTop()):
            sum = 0.0
            for phase_id in range(self.phase_num[None]):
                sum += self.obj.phase.val_frac[part_id, phase_id]
            if sum < fact:
                # print('empty phase', part_id, sum)
                self.obj.rgb[part_id] = WHITE
            if sum > 2-fact:
                # print('empty phase', part_id, sum)
                self.obj.rgb[part_id] = DARK
    
    @ti.kernel
    def debug_draw_lambda(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            self.obj.rgb[part_id] = ti.Vector([1, self.obj.mixture.lamb[part_id], self.obj.mixture.lamb[part_id]])

    @ti.kernel
    def debug_check_negative_phase(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            for phase_id in range(self.phase_num[None]):
                if self.obj.phase.val_frac[part_id, phase_id] < 0:
                    self.obj.rgb[part_id] = GREEN
                    print('negative phase', part_id, phase_id, self.obj.phase.val_frac[part_id, phase_id])
    
    @ti.kernel
    def debug_check_val_frac(self):
        sum_phase_1 = 0.0
        sum_phase_2 = 0.0
        sum_phase_3 = 0.0
        for part_id in range(self.obj.tiGetObjStackTop()):
            sum_phase_1 += self.obj.phase.val_frac[part_id, 0]
            sum_phase_2 += self.obj.phase.val_frac[part_id, 1]
            sum_phase_3 += self.obj.phase.val_frac[part_id, 2]
        print('phase 1 total', sum_phase_1)
        # print('phase 2 total', sum_phase_2)
        print('phase 3 total', sum_phase_3)

    @ti.kernel
    def debug_check_negative_phase(self):
        for part_id in range(self.obj.tiGetObjStackTop()):
            for phase_id in range(self.phase_num[None]):
                if self.obj.phase.val_frac[part_id, phase_id] < 0:
                    self.obj.rgb[part_id] = GREEN
                    print('negative phase', part_id, phase_id, self.obj.phase.val_frac[part_id, phase_id])

    @ti.kernel
    def debug_max_phase_vel(self) -> ti.f32:
        max_vel = 0.0
        for part_id in range(self.obj.tiGetObjStackTop()):
            for phase_id in range(self.phase_num[None]):
                ti.atomic_max(max_vel, ti.math.length(self.obj.phase.vel[part_id, phase_id]))
        return max_vel