import taichi as ti
import math
from .sph_funcs import *
from ..basic_obj.Obj_Particle import Particle
from .Solver import Solver

@ti.data_oriented
class SPH_solver(Solver):
    def __init__(self, obj: Particle):
        
        Solver.__init__(self, obj)

        self.dim = obj.m_world.g_dim
        self.compute_sig(self.sig_dim(self.getObj().getObjWorld().getWorldDim()))

        self.dt=obj.m_world.g_dt
        self.inv_dt = obj.m_world.g_inv_dt
        self.neg_inv_dt = obj.m_world.g_neg_inv_dt
        self.inv_dt2 = obj.m_world.g_inv_dt2

    @ti.kernel
    def loop_neighb(self, neighb_pool:ti.template(), neighb_obj:ti.template(), func:ti.template()):
        for part_id in range(self.tiGetObj().tiGetObjStackTop()):
            neighbPart_num = neighb_pool.tiGet_partNeighbObjSize(part_id, neighb_obj.tiGetObjId()[None])
            neighbPool_pointer = neighb_pool.tiGet_partNeighbObjBeginingPointer(part_id, neighb_obj.tiGetObjId()[None])
            for neighb_part_iter in range(neighbPart_num):
                neighbPart_id = neighb_pool.tiGet_neighbPartId(neighbPool_pointer)
                ''' Code for Computation'''
                func(part_id, neighbPart_id, neighbPool_pointer, neighb_pool, neighb_obj)
                ''' End of Code for Computation'''
                ''' DO NOT FORGET TO COPY/PASE THE FOLLOWING CODE WHEN REUSING THIS FUNCTION '''
                neighbPool_pointer = neighb_pool.tiGet_nextPointer(neighbPool_pointer)
    
    ''' [NOTICE] If sig_dim is decorated with @ti.func, and called in a kernel, 
    it will cause a computation error due to the use of math.pi. This bug is tested. '''
    def sig_dim(self, dim):
        sig = 0
        if dim == 3:
            sig = 8 / math.pi 
        elif dim == 2:
            sig = 40 / 7 / math.pi
        elif dim == 1:
            sig = 4 / 3
        return sig
    
    @ti.kernel
    def compute_sig(self, sig_dim: ti.f32):
        for part_id in range(self.tiGetObj().tiGetObjStackTop()):
            self.tiGetObj().tiSetSphH(part_id, self.tiGetObj().tiGetPartSize(part_id) * 2)
            self.tiGetObj().tiSetSphSig(part_id, sig_dim / ti.pow(self.tiGetObj().tiGetSphH(part_id), self.tiGetObj().tiGetObjWorld().tiGetWorldDim()))
            self.tiGetObj().tiSetSphSigInvH(part_id, self.tiGetObj().tiGetSphSig(part_id) / self.tiGetObj().tiGetSphH(part_id))

    @ti.func
    def inloop_accumulate_density(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_W = neighb_pool.tiGet_cachedW(neighb_part_shift)
        self.tiGetObj().tiAddSphDensity(part_id, neighb_obj.tiGetMass(neighb_part_id) * cached_W)
    
    @ti.func
    def inloop_accumulate_number_density(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_W = neighb_pool.tiGet_cachedW(neighb_part_shift)
        self.tiGetObj().tiAddSphDensity(part_id, self.tiGetObj().tiGetMass(part_id) * cached_W)

    @ti.func
    def inloop_accumulate_compression_ratio(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_W = neighb_pool.tiGet_cachedW(neighb_part_shift)
        self.tiGetObj().tiAddSphCompressionRatio(part_id, neighb_obj.tiGetVolume(neighb_part_id) * cached_W)
    
    @ti.func
    def inloop_avg_velocity(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_W = neighb_pool.tiGet_cachedW(neighb_part_shift)
        self.tiGetObj().tiAddVel(part_id, neighb_obj.tiGetVel(neighb_part_id) * neighb_obj.tiGetVolume(neighb_part_id) * cached_W)

    @ti.func
    def inloop_accumulate_strainRate(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        if bigger_than_zero(cached_dist):
            self.tiGetObj().tiAddStrainRate(part_id, neighb_obj.tiGetVolume(neighb_part_id)*(neighb_obj.tiGetVel(neighb_part_id)-self.tiGetObj().tiGetVel(part_id)).outer_product(cached_grad_W))

    @ti.func
    def inloop_avg_strainRate(self, part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
        cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
        cached_W = neighb_pool.tiGet_cachedW(neighb_part_shift)
        cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
        self.tiGetObj().tiAddStrainRate(part_id, neighb_obj.tiGetStrainRate(neighb_part_id) * neighb_obj.tiGetVolume(neighb_part_id) * cached_W)

    def sph_compute_density(self, neighb_pool):
        self.getObj().clear(self.getObj().getSphDensityArr())
        for neighb_obj in neighb_pool.neighb_obj_list:
            ''' Compute Density '''
            self.loop_neighb(neighb_pool, neighb_obj, self.inloop_accumulate_density)
    
    def sph_compute_number_density(self, neighb_pool):
        self.getObj().clear(self.getObj().getSphDensityArr())
        for neighb_obj in neighb_pool.neighb_obj_list:
            ''' Compute Density '''
            self.loop_neighb(neighb_pool, neighb_obj, self.inloop_accumulate_number_density)
    
    def sph_compute_compression_ratio(self, neighb_pool):
        self.getObj().clear(self.getObj().getSphCompressionRatioArr())
        for neighb_obj in neighb_pool.neighb_obj_list:
            ''' Compute Compression Ratio '''
            self.loop_neighb(neighb_pool, neighb_obj, self.inloop_accumulate_compression_ratio)
    
    def sph_avg_velocity(self, neighb_pool):
        self.getObj().clear(self.getObj().getVelArr())
        for neighb_obj in neighb_pool.neighb_obj_list:
            ''' Compute Average Velocity '''
            self.loop_neighb(neighb_pool, neighb_obj, self.inloop_avg_velocity)

    def sph_compute_strainRate(self, neighb_obj, neighb_pool):
        self.getObj().clear(self.getObj().getStrainRateArr())
        self.loop_neighb(neighb_pool, neighb_obj, self.inloop_accumulate_strainRate)
    
    def sph_avg_strainRate(self, neighb_pool):
        self.getObj().clear(self.getObj().getStrainRateArr())
        for neighb_obj in neighb_pool.neighb_obj_list:
            ''' Compute Strain Rate '''
            self.loop_neighb(neighb_pool, neighb_obj, self.inloop_avg_strainRate)
    
