# core of SPH (includes smoothing kernels, time step conditions, helper functions, SPH approximatons, and initializing smoothing kernel related particle attributes)

from numpy import float32
import taichi as ti
import math

INF_SMALL = 1e-6
REL_SMALL = 0.001

# ====================================================================================
# smoothing kernels

# FROM: Eqn.(2) of the paper "Versatile Surface Tension and Adhesion for SPH Fluids"
# REF: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.462.8293&rep=rep1&type=pdf
# NOTE: this func is insensitive to the $dim$
@ti.func
def spline_C(r, h):
    q = r / h
    tmp = 0.0
    if q <= 0.5:
        tmp = 2 * (1 - q) ** 3 * q**3 - 1 / 64
    elif q > 0.5 and q < 1:
        tmp = (1 - q) ** 3 * q**3
    tmp *= 32 / math.pi / h**3
    return tmp


# FROM: Eqn.(4) of the paper "Smoothed Particle Hydrodynamics Techniques for the Physics Based Simulation of Fluids and Solids"
# REF: https://github.com/InteractiveComputerGraphics/SPH-Tutorial/blob/master/pdf/SPH_Tutorial.pdf
# NOTE: $dim$ is implicitly defined in the param $sig$
@ti.func
def spline_W_old(r, h, sig):
    q = r / h
    tmp = 0.0
    if q <= 0.5:
        tmp = 6 * (q**3 - q**2) + 1
    elif q > 0.5 and q < 1:
        tmp = 2 * (1 - q) ** 3
    tmp *= sig
    return tmp


@ti.func
def spline_W(r, h, sig):
    q = r / h
    tmp = 0.0
    if 1> q > 0.5:
        tmp = 2 * (1 - q) ** 3
        tmp *= sig
    elif q <= 0.5:
        tmp = 6 * (q**3 - q**2) + 1
        tmp *= sig
    return tmp


# FROM: Eqn.(4) of the paper "Smoothed Particle Hydrodynamics Techniques for the Physics Based Simulation of Fluids and Solids"
# REF: https://github.com/InteractiveComputerGraphics/SPH-Tutorial/blob/master/pdf/SPH_Tutorial.pdf
# NOTE: This fun is spline_W() with the derivative of $r$
@ti.func
def grad_spline_W_old(r, h, sig):
    q = r / h
    tmp = 0.0
    if q <= 0.5:
        tmp = 6 * (3 * q**2 - 2 * q)
    elif q > 0.5 and q < 1:
        tmp = -6 * (1 - q) ** 2
    tmp *= sig / h
    return tmp


@ti.func
def grad_spline_W(r, h, sig_inv_h):
    q = r / h
    tmp = 0.0
    if 1> q > 0.5:
        tmp = -6 * (1 - q) ** 2
        tmp *= sig_inv_h
    elif q <= 0.5:
        tmp = 6 * (3 * q**2 - 2 * q)
        tmp *= sig_inv_h
    return tmp


# FROM: Eqn.(26) of the paper "Smoothed Particle Hydrodynamics Techniques for the Physics Based Simulation of Fluids and Solids"
# REF: https://github.com/InteractiveComputerGraphics/SPH-Tutorial/blob/master/pdf/SPH_Tutorial.pdf
# NOTE: x_ij and A_ij should be all Vector and be alinged
#       e.g. x_ij=ti.Vector([1,2]) A_ij=ti.Vector([1,2])
# NOTE: V_j is the volume of particle j, V_j==m_j/rho_j==Vj0/compression_rate_j
@ti.func
def artificial_Laplacian_spline_W(
    r, grad_W, dim, V_j, x_ij: ti.template(), A_ij: ti.template()
):
    return 2 * (2 + dim) * V_j * grad_W * (x_ij) * A_ij.dot(x_ij) / (r**3)

# ====================================================================================
# helper functions

@ti.func
def bigger_than_zero(val: ti.f32):
    if_bigger_than_zero = False
    if val > INF_SMALL:
        if_bigger_than_zero = True
    return if_bigger_than_zero


@ti.func
def make_bigger_than_zero():
    return INF_SMALL

# ====================================================================================
# decide timestep length

# fixed timestep length
@ti.kernel
def fixed_dt(cs: ti.f32, discretization_size: ti.f32, cfl_factor: ti.f32) -> ti.f32:
    return discretization_size / cs * cfl_factor


# compute output_dt and output_inv_dt with cfl condition
# what are min_acc_norm and acc_dt for?
@ti.kernel
def cfl_dt(
    obj: ti.template(),
    obj_size: ti.template(),
    obj_vel: ti.template(),
    cfl_factor: ti.template(),
    standard_dt: ti.f32,
    output_dt: ti.template(),
    output_inv_dt: ti.template(),
):
    dt = ti.Vector([100.0])
    dt[0] = 100

    for i in range(obj.info.stack_top[None]):
        vel_dt = obj_size[i] * cfl_factor[None]
        vel_norm = obj_vel[i].norm()
        if bigger_than_zero(vel_norm):
            vel_dt = obj_size[i] / vel_norm * cfl_factor[None]
        ti.atomic_min(dt[0], vel_dt)

    dt[0] = ti.min(dt[0], standard_dt)
    output_dt[None] = dt[0]


# @ti.data_oriented
# class SPH_kernel:
#     def __init__(self):
#         pass

#     # ====================================================================================
#     # SPH approximations

#     # output_attr += SPH approximation of input_attr (using volume)
#     @ti.kernel
#     def compute_W(
#         self,
#         obj: ti.template(),
#         nobj: ti.template(),
#         nobj_volume: ti.template(),
#         nobj_input_attr: ti.template(),
#         obj_output_attr: ti.template(),
#         config_neighb: ti.template(),
#     ):
#         cell_vec = ti.static(obj.located_cell.vec)
#         for i in range(obj.info.stack_top[None]):
#             for cell_tpl in range(config_neighb.search_template.shape[0]):
#                 cell_coded = (
#                     cell_vec[i] + config_neighb.search_template[cell_tpl]
#                 ).dot(config_neighb.cell_coder[None])
#                 if 0 < cell_coded < config_neighb.cell_num[None]:
#                     for j in range(nobj.cell.part_count[cell_coded]):
#                         shift = nobj.cell.part_shift[cell_coded] + j
#                         nid = nobj.located_cell.part_log[shift]
#                         """compute below"""
#                         dis = (obj.basic.pos[i], nobj.basic.pos[nid]).norm()
#                         obj_output_attr[i] += (
#                             nobj_input_attr[nid]
#                             * nobj_volume[nid]
#                             * spline_W(dis, obj.sph.h[i], obj.sph.sig[i])
#                         )

#     # output_attr += SPH gradient approximation of input_attr (using volume) (using neither difference nor symmetry equation)
#     @ti.kernel
#     def compute_W_grad(
#         self,
#         obj: ti.template(),
#         nobj: ti.template(),
#         nobj_volume: ti.template(),
#         nobj_input_attr: ti.template(),
#         obj_output_attr: ti.template(),
#         config_neighb: ti.template(),
#     ):
#         cell_vec = ti.static(obj.located_cell.vec)
#         for i in range(obj.info.stack_top[None]):
#             for cell_tpl in range(config_neighb.search_template.shape[0]):
#                 cell_coded = (
#                     cell_vec[i] + config_neighb.search_template[cell_tpl]
#                 ).dot(config_neighb.cell_coder[None])
#                 if 0 < cell_coded < config_neighb.cell_num[None]:
#                     for j in range(nobj.cell.part_count[cell_coded]):
#                         shift = nobj.cell.part_shift[cell_coded] + j
#                         nid = nobj.located_cell.part_log[shift]
#                         """compute below"""
#                         x_ij = obj.basic.pos[i] - nobj.basic.pos[nid]
#                         dis = x_ij.norm()
#                         if dis > INF_SMALL:
#                             grad_W_vec = (
#                                 grad_spline_W(dis, obj.sph.h[i], obj.sph.sig_inv_h[i])
#                                 * x_ij
#                                 / dis
#                             )
#                             obj_output_attr[i] += (
#                                 nobj_input_attr[nid] * nobj_volume[nid] * grad_W_vec
#                             )

#     # output_attr += SPH gradient approximation of input_attr (using volume) (using difference equation), typo: grad -> grand
#     @ti.kernel
#     def compute_W_grand_diff(
#         self,
#         obj: ti.template(),
#         nobj: ti.template(),
#         nobj_volume: ti.template(),
#         obj_input_attr: ti.template(),
#         nobj_input_attr: ti.template(),
#         obj_output_attr: ti.template(),
#         config_neighb: ti.template(),
#     ):
#         cell_vec = ti.static(obj.located_cell.vec)
#         for i in range(obj.info.stack_top[None]):
#             for cell_tpl in range(config_neighb.search_template.shape[0]):
#                 cell_coded = (
#                     cell_vec[i] + config_neighb.search_template[cell_tpl]
#                 ).dot(config_neighb.cell_coder[None])
#                 if 0 < cell_coded < config_neighb.cell_num[None]:
#                     for j in range(nobj.cell.part_count[cell_coded]):
#                         shift = nobj.cell.part_shift[cell_coded] + j
#                         nid = nobj.located_cell.part_log[shift]
#                         """compute below"""
#                         x_ij = obj.basic.pos[i] - nobj.basic.pos[nid]
#                         dis = x_ij.norm()
#                         if dis > INF_SMALL:
#                             grad_W_vec = (
#                                 grad_spline_W(dis, obj.sph.h[i], obj.sph.sig_inv_h[i])
#                                 * x_ij
#                                 / dis
#                             )
#                             obj_output_attr[i] += (
#                                 (nobj_input_attr[nid] - obj_input_attr[i])
#                                 * nobj_volume[nid]
#                                 * grad_W_vec
#                             )

#     # output_attr += SPH gradient approximation of input_attr (using volume) (using the A_i + A_j symmetric equation), typo: grad -> grand
#     @ti.kernel
#     def compute_W_grand_sum(
#         self,
#         obj: ti.template(),
#         nobj: ti.template(),
#         nobj_volume: ti.template(),
#         obj_input_attr: ti.template(),
#         nobj_input_attr: ti.template(),
#         obj_output_attr: ti.template(),
#         config_neighb: ti.template(),
#     ):
#         cell_vec = ti.static(obj.located_cell.vec)
#         for i in range(obj.info.stack_top[None]):
#             for cell_tpl in range(config_neighb.search_template.shape[0]):
#                 cell_coded = (
#                     cell_vec[i] + config_neighb.search_template[cell_tpl]
#                 ).dot(config_neighb.cell_coder[None])
#                 if 0 < cell_coded < config_neighb.cell_num[None]:
#                     for j in range(nobj.cell.part_count[cell_coded]):
#                         shift = nobj.cell.part_shift[cell_coded] + j
#                         nid = nobj.located_cell.part_log[shift]
#                         """compute below"""
#                         x_ij = obj.basic.pos[i] - nobj.basic.pos[nid]
#                         dis = x_ij.norm()
#                         if dis > INF_SMALL:
#                             grad_W_vec = (
#                                 grad_spline_W(dis, obj.sph.h[i], obj.sph.sig_inv_h[i])
#                                 * x_ij
#                                 / dis
#                             )
#                             obj_output_attr[i] += (
#                                 (nobj_input_attr[nid] + obj_input_attr[i])
#                                 * nobj_volume[nid]
#                                 * grad_W_vec
#                             )

#     # obj_output_attr += laplacian approximation of input_attr
#     @ti.kernel
#     def compute_Laplacian(
#         self,
#         obj: ti.template(),
#         obj_pos: ti.template(),
#         nobj_pos: ti.template(),
#         nobj_volume: ti.template(),
#         obj_input_attr: ti.template(),
#         nobj_input_attr: ti.template(),
#         coeff: ti.template(),
#         obj_output_attr: ti.template(),
#         background_neighb_grid: ti.template(),
#         search_template: ti.template(),
#     ):
#         dim = ti.static(obj.basic.pos[0].n)
#         for pid in range(obj.info.stack_top[None]):
#             located_cell = background_neighb_grid.get_located_cell(
#                 pos=obj_pos[pid],
#             )
#             for neighb_cell_iter in range(search_template.get_neighb_cell_num()):
#                 neighb_cell_index = background_neighb_grid.get_neighb_cell_index(
#                     located_cell=located_cell,
#                     cell_iter=neighb_cell_iter,
#                     neighb_search_template=search_template,
#                 )
#                 if background_neighb_grid.within_grid(neighb_cell_index):
#                     for neighb_part in range(
#                         background_neighb_grid.get_cell_part_num(neighb_cell_index)
#                     ):
#                         nid = background_neighb_grid.get_neighb_part_id(
#                             cell_index=neighb_cell_index,
#                             neighb_part_index=neighb_part,
#                         )
#                         """compute below"""
#                         x_ij = obj_pos[pid] - nobj_pos[nid]
#                         dis = x_ij.norm()
#                         if dis > INF_SMALL:
#                             grad_W = grad_spline_W(
#                                 dis, obj.sph.h[pid], obj.sph.sig_inv_h[pid]
#                             )
#                             A_ij = obj_input_attr[pid] - nobj_input_attr[nid]
#                             obj_output_attr[pid] += coeff[
#                                 None
#                             ] * artificial_Laplacian_spline_W(
#                                 dis,
#                                 grad_W,
#                                 dim,
#                                 nobj_volume[nid],
#                                 x_ij,
#                                 A_ij,
#                             )

#     # ====================================================================================
#     # initialize per-particle attributes that are related to smoothing kernel

#     # initialize support radius per-particle
#     @ti.kernel
#     def set_h(
#         self,
#         obj: ti.template(),
#         obj_output_h: ti.template(),
#         h: ti.f32,
#     ):
#         for i in range(obj.info.stack_top[None]):
#             obj_output_h[i] = h

#     # compute smoothing kernel normalization factor
#     def compute_sig(self, obj, obj_output_sig):
#         dim = ti.static(obj.basic.pos[0].n)
#         sig = 0
#         if dim == 3:
#             sig = 8 / math.pi
#         elif dim == 2:
#             sig = 40 / 7 / math.pi
#         elif dim == 1:
#             sig = 4 / 3
#         else:
#             print("Exception from compute_sig():")
#             print("dim out of range")
#             exit(0)
#         self.compute_sig_ker(obj, obj_output_sig, sig)

#     # compute smoothing kernel normalization factor (ti kernel)
#     @ti.kernel
#     def compute_sig_ker(
#         self, obj: ti.template(), obj_output_sig: ti.template(), sig: ti.f32
#     ):
#         dim = ti.static(obj.basic.pos[0].n)
#         for i in range(obj.info.stack_top[None]):
#             obj_output_sig[i] = sig / ti.pow(obj.sph.h[i], dim)

#     # compute sig/h
#     @ti.kernel
#     def compute_sig_inv_h(
#         self,
#         obj: ti.template(),
#         obj_sig: ti.template(),
#         obj_h: ti.template(),
#         obj_output_sig_inv_h: ti.template(),
#     ):
#         for i in range(obj.info.stack_top[None]):
#             obj_output_sig_inv_h[i] = obj_sig[i] / obj_h[i]

#     # a part of per-particle initialization (related to smoothing kernel)
#     def compute_kernel(
#         self,
#         obj,
#         h,
#         obj_output_h,
#         obj_output_sig,
#         obj_output_sig_inv_h,
#     ):
#         self.set_h(obj, obj_output_h, h)
#         self.compute_sig(obj, obj_output_sig)
#         self.compute_sig_inv_h(obj, obj_output_sig, obj_output_h, obj_output_sig_inv_h)

#     # initialize smoothing kernel
#     def kernel_init(self):
#         dim = ti.static(self.obj.basic.pos[0].n)
#         sig = 0
#         if dim == 3:
#             sig = 8 / math.pi
#         elif dim == 2:
#             sig = 40 / 7 / math.pi
#         elif dim == 1:
#             sig = 4 / 3
#         else:
#             print("Exception from kernel_init():")
#             print("dim out of range")
#             exit(0)
#         self.kernel_init_h_and_sig(dim=dim, sig=sig)

#     @ti.kernel
#     def kernel_init_h_and_sig(
#         self,
#         dim: ti.i32,
#         sig: ti.f32,
#     ):
#         for pid in range(self.obj.info.stack_top[None]):
#             self.obj.sph.h[pid] = self.obj.basic.size[pid] * 2
#             self.obj.sph.sig[pid] = sig / ti.pow(self.obj.sph.h[pid], dim)
#             self.obj.sph.sig_inv_h[pid] = self.obj.sph.sig[pid] / self.obj.sph.h[pid]

#     @ti.kernel
#     def time_integral_arr(
#         self,
#         obj_frac: ti.template(),
#         obj_output_int: ti.template(),
#     ):
#         for i in range(self.obj.info.stack_top[None]):
#             obj_output_int[i] += obj_frac[i] * self.dt

#     @ti.kernel
#     def time_integral_val(
#         self,
#         obj_frac: ti.template(),
#         obj_output_int: ti.template(),
#     ):
#         for i in range(self.obj.info.stack_top[None]):
#             obj_output_int[i] += obj_frac[None] * self.dt

#     # update acceleration from force
#     @ti.kernel
#     def update_acc(
#         self,
#         obj: ti.template(),
#         obj_mass: ti.template(),
#         obj_force: ti.template(),
#         obj_output_acc: ti.template(),
#     ):
#         for i in range(obj.info.stack_top[None]):
#             obj_output_acc[i] += obj_force[i] / obj_mass[i]
