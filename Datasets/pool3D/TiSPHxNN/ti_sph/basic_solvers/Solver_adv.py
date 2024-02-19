import taichi as ti
from.sph_funcs import *
from .Solver_sph import Solver
from ..basic_op.type import *
from ..basic_obj.Obj_Particle import Particle

@ti.data_oriented
class Adv_slover(Solver):
    def __init__(self, obj: Particle):

        Solver.__init__(self, obj)
        self.clean_acc = vecxf(self.getObj().getObjWorld().getWorldDim())(0)
        self.chche_1 = vecxf(self.getObj().getObjWorld().getWorldDim())(0)
    
    @ti.kernel
    def clear_acc(self):
        for i in range(self.tiGetObj().tiGetObjStackTop()):
            self.tiGetObj().tiSetAcc(i, self.clean_acc)

    @ti.kernel
    def add_gravity_acc(self):
        for i in range(self.tiGetObj().tiGetObjStackTop()):
            self.tiGetObj().tiAddAcc(i, self.tiGetObj().tiGetObjWorld().tiGetWorldGravity())
    
    @ti.func
    def inloop_accumulate_vis_acc(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            k_vis = (self.tiGetObj().tiGetKVis(part_id) + neighb_obj.tiGetKVis(neighb_part_id)) / 2
            A_ij = self.tiGetObj().tiGetVel(part_id) - neighb_obj.tiGetVel(neighb_part_id)
            x_ij = self.tiGetObj().tiGetPos(part_id) - neighb_obj.tiGetPos(neighb_part_id)
            self.tiGetObj().tiAddAcc(part_id, k_vis*2*(2+self.tiGetObj().tiGetObjWorld().tiGetWorldDim()) * neighb_obj.tiGetVolume(neighb_part_id) * cached_grad_W * A_ij.dot(x_ij) / (cached_dist**2))
    
    @ti.kernel
    def add_acc_gravity(self):
        for i in range(self.tiGetObj().tiGetObjStackTop()):
            self.tiGetObj().tiAddAcc(i, self.tiGetObj().tiGetObjWorld().tiGetWorldGravity())

    '''
    harmonic acceleration
    '''
    @ti.kernel
    def add_acc_harmonic(self, axis: ti.i32, dir: ti.f32, period: ti.f32, amplitude: ti.f32, world_time: ti.f32, delay: ti.f32):

        time = world_time - delay
        if time > 0:
            acc = self.chche_1
            acc.fill(0)
            
            angular_frequency = ti.math.pi * 2 / period
            acc[axis] = -amplitude * (angular_frequency ** 2) * ti.math.cos(angular_frequency * time) * dir
            
            for i in range(self.tiGetObj().tiGetObjStackTop()):
                self.tiGetObj().tiAddAcc(i, acc)


    @ti.kernel
    def acc2vel_adv(self):
        for i in range(self.tiGetObj().tiGetObjStackTop()):
            self.tiGetObj().tiSetVelAdv(i, self.tiGetObj().tiGetAcc(i) * self.tiGetObj().tiGetObjWorld().tiGetWorldDt() + self.tiGetObj().tiGetVel(i))
    
    @ti.kernel
    def acc2vel(self):
        for i in range(self.tiGetObj().tiGetObjStackTop()):
            self.tiGetObj().tiSetVel(i, self.tiGetObj().tiGetAcc(i) * self.tiGetObj().tiGetObjWorld().tiGetWorldDt() + self.tiGetObj().tiGetVel(i))

    @ti.kernel
    def vel_adv2vel(self):
        for i in range(self.tiGetObj().tiGetObjStackTop()):
            self.tiGetObj().tiSetVel(i, self.tiGetObj().tiGetVelAdv(i))

    @ti.kernel
    def update_pos(self):
        for i in range(self.tiGetObj().tiGetObjStackTop()):
            self.tiGetObj().tiAddPos(i, self.tiGetObj().tiGetVel(i) * self.tiGetObj().tiGetObjWorld().tiGetWorldDt())

    @ti.kernel
    def adv_step(self, in_vel: ti.template(), out_vel_adv: ti.template()):
        for i in range(self.tiGetObj().tiGetObjStackTop()):
            out_vel_adv[i] = in_vel[i]
            self.tiGetObj().tiSetAcc(i, self.clean_acc)
            self.tiGetObj().tiAddAcc(i, self.tiGetObj().tiGetObjWorld().tiGetWorldGravity())
            out_vel_adv[i] += self.tiGetObj().tiGetAcc(i) * self.tiGetObj().tiGetObjWorld().tiGetWorldDt()