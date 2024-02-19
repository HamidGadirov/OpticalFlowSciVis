import taichi as ti
# from ti_sph import *
import ti_sph as tsph
from template_part import part_template
import time
import sys
import numpy as np
import csv
np.set_printoptions(threshold=sys.maxsize)

@ti.kernel
def clear_selected(fluid_part:ti.template()):
    for i in range(fluid_part.tiGetObjStackTop()):
        fluid_part.selected[i] = 0

@ti.kernel
def select_part(fluid_part:ti.template()):
    for i in range(fluid_part.tiGetObjStackTop()):
        if fluid_part.sph.compression_ratio[i] < 0.9:
            fluid_part.selected[i] = 1
        else:
            fluid_part.selected[i] = 0

@ti.func
def inloop_expand_selected(part_id: ti.i32, neighb_part_id: ti.i32, neighb_part_shift: ti.i32, neighb_pool:ti.template(), neighb_obj:ti.template()):
    cached_dist = neighb_pool.tiGet_cachedDist(neighb_part_shift)
    cached_grad_W = neighb_pool.tiGet_cachedGradW(neighb_part_shift)
    if cached_dist>1e-6 and fluid_part.selected[part_id]==1 and neighb_obj.selected[neighb_part_id] != 1:
        neighb_obj.selected[neighb_part_id] = 2

@ti.kernel
def confirm_selected(fluid_part:ti.template()):
    for i in range(fluid_part.tiGetObjStackTop()):
        if fluid_part.selected[i] == 2:
            fluid_part.selected[i] = 1

@ti.kernel
def color_selected(fluid_part:ti.template()):
    for i in range(fluid_part.tiGetObjStackTop()):
        if fluid_part.selected[i] == 1:
            fluid_part.rgb[i] = tsph.vec3f(1,0,0)
        else:
            fluid_part.rgb[i] = tsph.vec3f(0.9,0.9,0.9)

''' TAICHI SETTINGS '''
# Use GPU, comment the below command to run this programme on CPU
ti.init(arch=ti.cuda, device_memory_GB=13) 
# Use CPU, uncomment the below command to run this programme if you don't have GPU
# ti.init(arch=ti.vulkan)
# ti.init(arch=ti.cpu)

''' SOLVER SETTINGS '''
SOLVER_ISM = 0  # proposed method
SOLVER_JL21 = 1 # baseline method
solver = SOLVER_ISM # choose the solver

''' SETTINGS OUTPUT DATA '''
# output fps
fps = 60
# max output frame number
output_frame_num = 2000

''' SETTINGS SIMULATION '''
# size of the particle
part_size = 0.005 
# part_size = 0.01 
dpi=800
# number of phases
phase_num = 3 
# max time step size
if solver == SOLVER_ISM:
    max_time_step = part_size/100
elif solver == SOLVER_JL21:
    max_time_step = part_size/100
#  diffusion amount: Cf = 0 yields no diffusion
Cf = 0.0 
#  drift amount (for ism): Cd = 0 yields free driftand Cd = 1 yields no drift
Cd = 0.8 
# drag coefficient (for JL21): kd = 0 yields maximum drift 
kd = 0.0
flag_strat_drift = True
# kinematic viscosity of fluid
kinematic_viscosity_fluid = 1e-3

''' INIT SIMULATION WORLD '''
# create a 2D world
world = tsph.World(dim=2) 
# set the particle diameter
world.setWorldPartSize(part_size) 
# set the max time step size
world.setWorldDt(max_time_step) 
# set up the multiphase. The first argument is the number of phases. The second argument is the color of each phase (RGB). The third argument is the rest density of each phase.
world.set_multiphase(phase_num,[tsph.vec3f(0.0,0.2,0.8),tsph.vec3f(0.8,0.2,0.0),tsph.vec3f(0.8,0.2,0.0)],[300,500,1000]) 

''' DATA SETTINGS FOR FLUID PARTICLE '''
# generate the fluid particle data as a hollowed sphere, rotating irrotationally
pool_data = tsph.Squared_pool_2D_data(container_height=8, container_size=5, fluid_height=7, span=world.g_part_size[None]*1.0005, layer = 3)
# water_data = tsph.Cube_data(span=world.g_part_size[None], type=tsph.Cube_data.FIXED_CELL_SIZE, lb=tsph.vec3f(-0.6,0.0,-0.6), rt=tsph.vec3f(0.6,3.5,0.6))
# glass_data = tsph.Ply_data('./glassPoint.ply')
# particle number of fluid/boundary
fluid_part_num = pool_data.fluid_part_num
bound_part_num = pool_data.bound_part_num
print("fluid_part_num", fluid_part_num)
# position info of fluid/boundary (as numpy arrays)
fluid_part_pos = pool_data.fluid_part_pos
bound_part_pos = pool_data.bound_part_pos
# initial velocity info of fluid

'''INIT AN FLUID PARTICLE OBJECT'''
# create a fluid particle object. first argument is the number of particles. second argument is the size of the particle. third argument is whether the particle is dynamic or not.
fluid_part = tsph.Particle(part_num=fluid_part_num, part_size=tsph.val_f(world.getWorldPartSize()), is_dynamic=True)
world.attachPartObj(fluid_part)
fluid_part.instantiate_from_template(part_template, world)

''' FEED DATA TO THE FLUID PARTICLE OBJECT '''
fluid_part.open_stack(fluid_part_num) # open the stack to feed data
fluid_part.fill_open_stack_with_nparr(fluid_part.pos, fluid_part_pos) # feed the position data
fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.getObjPartSize()) # feed the particle size
fluid_part.fill_open_stack_with_val(fluid_part.volume, fluid_part.getObjPartSize()**world.getWorldDim()) # feed the particle volume
val_frac = ti.field(dtype=ti.f32, shape=phase_num) # create a field to store the volume fraction
val_frac[0], val_frac[1], val_frac[2] = 1.0,0.0,0.0 # set up the volume fraction
fluid_part.fill_open_stack_with_vals(fluid_part.phase.val_frac, val_frac) # feed the volume fraction
fluid_part.close_stack() # close the stack

''' INIT A BOUNDARY PARTICLE OBJECT '''
bound_part = tsph.Particle(part_num=bound_part_num, part_size=world.g_part_size, is_dynamic=False)
world.attachPartObj(bound_part)
bound_part.instantiate_from_template(part_template, world)

''' FEED DATA TO THE BOUNDARY PARTICLE OBJECT '''
bound_part.open_stack(bound_part_num)
bound_part.fill_open_stack_with_nparr(bound_part.pos, bound_part_pos)
bound_part.fill_open_stack_with_val(bound_part.size, bound_part.getObjPartSize())
bound_part.fill_open_stack_with_val(bound_part.volume, bound_part.getObjPartSize()**world.getWorldDim())
bound_part.fill_open_stack_with_val(bound_part.mass, 1000*bound_part.getObjPartSize()**world.getWorldDim())
bound_part.fill_open_stack_with_val(bound_part.rest_density, 1000)
bound_part.close_stack()


'''INIT NEIGHBOR SEARCH OBJECTS'''
neighb_list=[fluid_part, bound_part]
fluid_part.add_module_neighb_search()
bound_part.add_module_neighb_search()

fluid_part.add_neighb_objs(neighb_list)
bound_part.add_neighb_objs(neighb_list)

fluid_part.add_solver_adv()
fluid_part.add_solver_sph()
fluid_part.add_solver_df(div_free_threshold=2e-4, incomp_warm_start=False, div_warm_start=False)
fluid_part.add_solver_ism(Cd=Cd, Cf=Cf, k_vis_inter=kinematic_viscosity_fluid, k_vis_inner=kinematic_viscosity_fluid)

bound_part.add_solver_sph()
bound_part.add_solver_df(div_free_threshold=2e-4)
bound_part.add_solver_ism(Cd=Cd, Cf=Cf, k_vis_inter=kinematic_viscosity_fluid, k_vis_inner=kinematic_viscosity_fluid)


''' INIT ALL SOLVERS '''
world.init_modules()

''' DATA PREPERATIONS '''
def prep_ism():
    world.neighb_search() # perform the neighbor search
    fluid_part.m_solver_ism.update_rest_density_and_mass()
    fluid_part.m_solver_ism.update_color() # update the color
    fluid_part.m_solver_ism.recover_phase_vel_from_mixture() # recover the phase velocity from the mixture velocity

''' SIMULATION LOOPS '''
def loop_ism():
    ''' color '''
    fluid_part.m_solver_ism.update_color()

    ''' neighb search'''
    ''' [TIME START] neighb_search '''
    world.neighb_search()
    ''' [TIME END] neighb_search '''

    ''' sph pre-computation '''
    ''' [TIME START] DFSPH Part 1 '''
    world.step_sph_compute_compression_ratio()
    world.step_df_compute_beta()
    ''' [TIME START] DFSPH Part 1 '''

    ''' pressure accleration (divergence-free) '''
    ''' [TIME START] DFSPH Part 2 '''
    world.step_vfsph_div(update_vel=False)
    ''' [TIME END] DFSPH Part 2 '''
    # print('div_free iter:', fluid_part.m_solver_df.div_free_iter[None])

    ''' [ISM] distribute pressure acc to phase acc and update phase vel '''
    '''  [TIME START] ISM Part 1 '''
    fluid_part.m_solver_df.get_acc_pressure()
    fluid_part.m_solver_ism.clear_phase_acc()
    fluid_part.m_solver_ism.ditribute_acc_pressure_2_phase()
    fluid_part.m_solver_ism.phase_acc_2_phase_vel()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()
    '''  [TIME END] ISM Part 1 '''
    

    ''' viscosity & gravity (not requird in this scene)  accleration and update phase vel '''
    '''  [TIME START] ISM Part 2 '''
    fluid_part.m_solver_ism.clear_phase_acc()
    fluid_part.m_solver_ism.add_phase_acc_gravity()
    fluid_part.m_solver_ism.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_ism.inloop_add_phase_acc_vis)
    fluid_part.m_solver_ism.phase_acc_2_phase_vel() 
    fluid_part.m_solver_ism.update_vel_from_phase_vel()
    '''  [TIME END] ISM Part 2 '''

    ''' pressure accleration (divergence-free) '''
    '''  [TIME START] DFSPH Part 3 '''
    world.step_vfsph_incomp(update_vel=False)
    '''  [TIME START] DFSPH Part 3 '''
    # print('incomp iter:', fluid_part.m_solver_df.incompressible_iter[None])

    ''' distribute pressure acc to phase acc and update phase vel '''
    '''  [TIME START] ISM Part 3 '''
    fluid_part.m_solver_df.get_acc_pressure()
    fluid_part.m_solver_ism.clear_phase_acc()
    fluid_part.m_solver_ism.ditribute_acc_pressure_2_phase()
    fluid_part.m_solver_ism.phase_acc_2_phase_vel()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()
    '''  [TIME END] ISM Part 3 '''

    ''' update particle position from velocity '''
    '''  [TIME START] DFSPH Part 4 '''
    world.update_pos_from_vel()
    '''  [TIME START] DFSPH Part 4 '''

    ''' phase change '''
    '''  [TIME START] ISM Part 4 '''
    fluid_part.m_solver_ism.update_val_frac()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()

    ''' update mass and velocity '''
    fluid_part.m_solver_ism.regularize_val_frac()
    fluid_part.m_solver_ism.update_rest_density_and_mass()
    fluid_part.m_solver_ism.update_vel_from_phase_vel()
    '''  [TIME END] ISM Part 4 '''

    ''' cfl condition update'''
    '''  [TIME START] CFL '''
    world.cfl_dt(0.4, max_time_step)
    '''  [TIME END] CFL '''

    ''' statistical info '''
    # print(' ')
    # fluid_part.m_solver_ism.statistics_linear_momentum_and_kinetic_energy()
    # fluid_part.m_solver_ism.statistics_angular_momentum()
    # fluid_part.m_solver_ism.debug_check_val_frac()

def write_part_info_ply():
    for part_id in range(fluid_part.getObjStackTop()):
        fluid_part.pos[part_id]
        fluid_part.vel[part_id]
        for phase_id in range(phase_num):
            fluid_part.phase.val_frac[part_id, phase_id]
        fluid_part.rgb[part_id]

    for bound_part_id in range(bound_part.getObjStackTop()):
        bound_part.pos[bound_part_id]

''' Viusalization and run '''
def run(loop):
    inv_fps = 1/fps
    timer = int(0)
    sim_time = float(0.0)
    loop_count = int(0)

    gui2d_part = tsph.Gui2d(objs=[fluid_part, bound_part], radius=world.getWorldPartSize()*8, lb=tsph.vec2f([-4,-4]),rt=tsph.vec2f([4,4]), dpi=dpi)

    while timer < output_frame_num:
        loop()
        loop_count += 1
        sim_time += world.getWorldDt()

        if(sim_time > timer*inv_fps):
            clear_selected(fluid_part)
            select_part(fluid_part)
            for i in range(40):
                fluid_part.m_solver_sph.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, inloop_expand_selected)
                confirm_selected(fluid_part)
            color_selected(fluid_part)
            

            gui2d_part.save_img(path='./output/part_'+str(timer)+'.jpg')
            timer += 1


''' RUN THE SIMULATION '''

prep_ism()
run(loop_ism)








