import taichi as ti
# from ti_sph import *
import ti_sph as tsph
from template_part import part_template
import time
import sys
import numpy as np
import csv
np.set_printoptions(threshold=sys.maxsize)

''' TAICHI SETTINGS '''
# Use GPU, comment the below command to run this programme on CPU
# ti.init(arch=ti.cuda, device_memory_GB=13) 
# Use CPU, uncomment the below command to run this programme if you don't have GPU
ti.init(arch=ti.cuda)
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
part_size = 0.05 
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
world = tsph.World(dim=3) 
# set the particle diameter
world.setWorldPartSize(part_size) 
# set the max time step size
world.setWorldDt(max_time_step) 
# set up the multiphase. The first argument is the number of phases. The second argument is the color of each phase (RGB). The third argument is the rest density of each phase.
world.set_multiphase(phase_num,[tsph.vec3f(0.8,0.2,0),tsph.vec3f(0,0.8,0.2),tsph.vec3f(0,0,1)],[500,500,1000]) 

''' DATA SETTINGS FOR FLUID PARTICLE '''
# generate the fluid particle data as a hollowed sphere, rotating irrotationally
pool_data = tsph.Squared_pool_3D_data(container_height=3, container_size=4, fluid_height=2, span=world.g_part_size[None], layer = 3)
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
fluid_part.fill_open_stack_with_val(fluid_part.volume, fluid_part.getObjPartSize()**world.g_dim[None]) # feed the particle volume
val_frac = ti.field(dtype=ti.f32, shape=phase_num) # create a field to store the volume fraction
val_frac[0], val_frac[1], val_frac[2] = 0.5,0.0,0.5 # set up the volume fraction
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
bound_part.fill_open_stack_with_val(bound_part.volume, bound_part.getObjPartSize()**world.g_dim[None])
bound_part.fill_open_stack_with_val(bound_part.mass, 1000*bound_part.getObjPartSize()**world.g_dim[None])
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
if solver == SOLVER_ISM:
    fluid_part.add_solver_df(div_free_threshold=2e-4, incomp_warm_start=False, div_warm_start=False)
    fluid_part.add_solver_ism(Cd=Cd, Cf=Cf, k_vis_inter=kinematic_viscosity_fluid, k_vis_inner=kinematic_viscosity_fluid)
elif solver == SOLVER_JL21:
    fluid_part.add_solver_wcsph()
    fluid_part.add_solver_JL21(kd=kd,Cf=Cf,k_vis=kinematic_viscosity_fluid)

bound_part.add_solver_sph()
if solver == SOLVER_ISM:
    bound_part.add_solver_df(div_free_threshold=2e-4)
    bound_part.add_solver_ism(Cd=Cd, Cf=Cf, k_vis_inter=kinematic_viscosity_fluid, k_vis_inner=kinematic_viscosity_fluid)
elif solver == SOLVER_JL21:
    bound_part.add_solver_wcsph()
    bound_part.add_solver_JL21(kd=kd,Cf=Cf,k_vis=kinematic_viscosity_fluid)

''' INIT ALL SOLVERS '''
world.init_modules()

''' DATA PREPERATIONS '''
def prep_ism():
    world.neighb_search() # perform the neighbor search
    fluid_part.m_solver_ism.update_rest_density_and_mass()
    fluid_part.m_solver_ism.update_color() # update the color
    fluid_part.m_solver_ism.recover_phase_vel_from_mixture() # recover the phase velocity from the mixture velocity

def prep_JL21():
    world.neighb_search() # perform the neighbor search
    fluid_part.m_solver_JL21.update_rest_density_and_mass()
    fluid_part.m_solver_JL21.update_color() # update the color
    fluid_part.m_solver_JL21.recover_phase_vel_from_mixture() # recover the phase velocity from the mixture velocity

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
    print('div_free iter:', fluid_part.m_solver_df.div_free_iter[None])

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
    print('incomp iter:', fluid_part.m_solver_df.incompressible_iter[None])

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
    print(' ')
    # fluid_part.m_solver_ism.statistics_linear_momentum_and_kinetic_energy()
    # fluid_part.m_solver_ism.statistics_angular_momentum()
    # fluid_part.m_solver_ism.debug_check_val_frac()

def loop_JL21():
    ''' update color based on the volume fraction '''
    fluid_part.m_solver_JL21.update_color()

    ''' neighb search'''
    ''' [TIME START] neighb_search '''
    world.neighb_search()
    ''' [TIME END] neighb_search '''

    ''' compute number density '''
    ''' [TIME START] WCSPH Part 1 '''
    world.step_sph_compute_number_density()
    ''' [TIME END] WCSPH Part 1 '''

    ''' gravity (not requird in this scene) accleration '''
    ''' [TIME START] WCSPH Part 2 '''
    world.clear_acc()
    world.add_acc_gravity()

    ''' viscosity force '''
    fluid_part.m_solver_JL21.clear_vis_force()
    fluid_part.m_solver_JL21.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_JL21.inloop_add_force_vis)
    fluid_part.m_solver_JL21.loop_neighb(fluid_part.m_neighb_search.neighb_pool, bound_part, fluid_part.m_solver_JL21.inloop_add_force_vis)

    ''' pressure force '''
    fluid_part.m_solver_JL21.clear_pressure_force()
    world.step_wcsph_add_acc_number_density_pressure()
    fluid_part.m_solver_JL21.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_JL21.inloop_add_force_pressure)
    fluid_part.m_solver_JL21.loop_neighb(fluid_part.m_neighb_search.neighb_pool, bound_part, fluid_part.m_solver_JL21.inloop_add_force_pressure)
    ''' [TIME END] WCSPH Part 2 '''

    ''' update phase vel (from all accelerations) '''
    if flag_strat_drift:
        ''' [TIME START] JL21 Part 1 '''
        fluid_part.m_solver_JL21.update_phase_vel()
        ''' [TIME END] JL21 Part 1 '''
    else:
        fluid_part.m_solver_JL21.vis_force_2_acc()
        fluid_part.m_solver_JL21.pressure_force_2_acc()
        fluid_part.m_solver_JL21.acc_2_vel()

    ''' update particle position from velocity '''
    ''' [TIME START] WCSPH Part 3 '''
    world.update_pos_from_vel()
    ''' [TIME END] WCSPH Part 3 '''

    ''' phase change (spacial care with lambda scheme) '''
    ''' [TIME START] JL21 Part 2 '''
    fluid_part.m_solver_JL21.update_val_frac_lamb()
    ''' [TIME END] JL21 Part 2 '''    

    ''' statistical info '''
    # print(' ')
    # fluid_part.m_solver_JL21.statistics_linear_momentum_and_kinetic_energy()
    # fluid_part.m_solver_JL21.statistics_angular_momentum()
    # fluid_part.m_solver_JL21.debug_check_val_frac()

    dt, max_vec = world.get_cfl_dt_obj(fluid_part, 0.5, max_time_step)
    world.setWorldDt(dt)    

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
def vis_run(loop):
    global flag_strat_drift
    inv_fps = 1/fps
    timer = 0
    sim_time = 0
    loop_count = 0
    flag_write_img = False

    gui = tsph.Gui3d()
    while gui.window.running:

        gui.monitor_listen()

        if gui.op_system_run:
            loop()
            loop_count += 1
            sim_time += world.g_dt[None]
            print("loop ", loop_count)
            
            if(sim_time > timer*inv_fps):
                if gui.op_write_file:
                    write_part_info_ply()
                timer += 1
                flag_write_img = True
        if gui.op_refresh_window:
            gui.scene_setup()
            gui.scene_add_parts_colorful(obj_pos=fluid_part.pos, obj_color=fluid_part.rgb,index_count=fluid_part.getObjStackTop(),size=world.g_part_size[None])
            # gui.scene_add_parts(obj_pos=bound_part.pos, obj_color=(0,0.5,1),index_count=bound_part.getObjStackTop(),size=world.g_part_size[None])
            gui.canvas.scene(gui.scene)  # Render the scene

            if gui.op_save_img and flag_write_img:
                gui.window.save_image('output/'+str(timer)+'.png')
                flag_write_img = False

            gui.window.show()
        
        if timer > output_frame_num:
            break

''' RUN THE SIMULATION '''
if solver == SOLVER_ISM:
    prep_ism()
    vis_run(loop_ism)
elif solver == SOLVER_JL21:
    prep_JL21()
    vis_run(loop_JL21)







