import taichi as ti
# from ti_sph import *
import ti_sph as tsph
from template_part import part_template
from template_grid import grid_template
import time
import sys
import numpy as np
import csv
np.set_printoptions(threshold=sys.maxsize)

output_path = '../output'

''' TAICHI SETTINGS '''
# ti.init(arch=ti.cuda, device_memory_GB=15)
ti.init(arch=ti.vulkan)

''' SETTINGS OUTPUT DATA '''
# output fps
fps = 60
# max output frame number
output_frame_num = 2000

''' SETTINGS SIMULATION '''
scaling_factor = 2
# size of the particle
part_size = 0.05 
# max time step size
max_time_step = part_size/100
sense_cell_size = part_size*2.5
# kinematic viscosity of fluid
kinematic_viscosity_fluid = 1e-4

''' INIT SIMULATION WORLD '''
# create a 3D world
world = tsph.World(dim=3) 
# set the particle diameter
world.setWorldPartSize(part_size) 
# set the max time step size
world.setWorldDt(max_time_step) 

''' DATA SETTINGS FOR FLUID PARTICLE '''
# generate the fluid particle data as a hollowed sphere, rotating irrotationally
fluid_cube_data = tsph.Cube_data(type=tsph.Cube_data.FIXED_CELL_SIZE, lb=tsph.vec3f(-0.5,-0.5,-0.5)*scaling_factor, rt=tsph.vec3f(0.5,1.5,0.5)*scaling_factor, span=world.getPartSize())
pool_data = tsph.Squared_pool_3D_data(container_height=6*scaling_factor, container_size=4*scaling_factor, fluid_height=2*scaling_factor, span=world.getPartSize(), layer = 3)
sense_cube_data = tsph.Cube_data(type=tsph.Cube_data.FIXED_GRID_RES, span=sense_cell_size, grid_res=tsph.vec3i(64,64,64),grid_center=tsph.vec3f(0,-2,0))
# particle number of fluid/boundary
fluid_part_num = pool_data.fluid_part_num + fluid_cube_data.num
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
fluid_part.open_stack(pool_data.fluid_part_num) # open the stack to feed data
fluid_part.fill_open_stack_with_nparr(fluid_part.pos, fluid_part_pos) # feed the position data
fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.getObjPartSize()) # feed the particle size
fluid_part.fill_open_stack_with_val(fluid_part.volume, fluid_part.getObjPartSize()**world.getWorldDim()) # feed the particle volume
fluid_part.fill_open_stack_with_val(fluid_part.mass, 1000*fluid_part.getObjPartSize()**world.getWorldDim()) # feed the particle mass
fluid_part.fill_open_stack_with_val(fluid_part.rest_density, 1000) # feed the particle rest density
fluid_part.fill_open_stack_with_val(fluid_part.rgb, tsph.vec3f([0.0, 0.0, 1.0]))
fluid_part.fill_open_stack_with_val(fluid_part.k_vis, kinematic_viscosity_fluid)
fluid_part.close_stack() # close the stack

fluid_part.open_stack(fluid_cube_data.num)
fluid_part.fill_open_stack_with_nparr(fluid_part.pos, fluid_cube_data.pos)
fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.getObjPartSize())
fluid_part.fill_open_stack_with_val(fluid_part.volume, fluid_part.getObjPartSize()**world.getWorldDim())
fluid_part.fill_open_stack_with_val(fluid_part.mass, 1000*fluid_part.getObjPartSize()**world.getWorldDim())
fluid_part.fill_open_stack_with_val(fluid_part.rest_density, 1000)
fluid_part.fill_open_stack_with_val(fluid_part.rgb, tsph.vec3f([0.0, 0.0, 1.0]))
fluid_part.fill_open_stack_with_val(fluid_part.k_vis, kinematic_viscosity_fluid)
fluid_part.close_stack()

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
bound_part.fill_open_stack_with_val(bound_part.rgb, tsph.vec3f([0.0, 0.5, 1.0]))
bound_part.fill_open_stack_with_val(bound_part.k_vis, kinematic_viscosity_fluid)
bound_part.close_stack()

sense_grid_part = tsph.Particle(part_num=sense_cube_data.num, part_size=tsph.val_f(sense_cell_size), is_dynamic=False)
world.attachPartObj(sense_grid_part)
sense_grid_part.instantiate_from_template(grid_template, world)
sense_grid_part.open_stack(sense_cube_data.num)
sense_grid_part.fill_open_stack_with_nparr(sense_grid_part.pos, sense_cube_data.pos)
sense_grid_part.fill_open_stack_with_nparr(sense_grid_part.node_index, sense_cube_data.index)
sense_grid_part.fill_open_stack_with_val(sense_grid_part.size, sense_grid_part.getObjPartSize())
sense_grid_part.fill_open_stack_with_val(sense_grid_part.volume, sense_grid_part.getObjPartSize()**world.getWorldDim())
sense_grid_part.close_stack()

'''INIT NEIGHBOR SEARCH OBJECTS'''
neighb_list=[fluid_part, bound_part]
fluid_part.add_module_neighb_search()
bound_part.add_module_neighb_search()
sense_grid_part.add_module_neighb_search(max_neighb_num=tsph.val_i(fluid_part.getObjPartNum()*48))

fluid_part.add_neighb_objs(neighb_list)
bound_part.add_neighb_objs(neighb_list)
sense_grid_part.add_neighb_obj(neighb_obj=fluid_part, search_range=tsph.val_f(sense_cell_size*2))

fluid_part.add_solver_adv()
fluid_part.add_solver_sph()
fluid_part.add_solver_df(div_free_threshold=1e-4, incomp_warm_start=True, div_warm_start=False)

bound_part.add_solver_sph()
bound_part.add_solver_df(div_free_threshold=1e-4)

sense_grid_part.add_solver_sph()

''' INIT ALL SOLVERS '''
world.init_modules()

world.neighb_search()

sense_output = tsph.Output_manager(format_type = tsph.Output_manager.type.SEQ, data_source = sense_grid_part)
sense_output.add_output_dataType("pos")
sense_output.add_output_dataType("sensed_density")
sense_output.add_output_dataType("vel")
sense_output.add_output_dataType("strainRate")
sense_output.add_one_time_output_dataType("node_index")

def prep():
    sense_output.export_one_time_to_numpy(path=output_path, compressed=True)

''' SIMULATION LOOPS '''
def loop_ism():

    ''' neighb search'''
    ''' [TIME START] neighb_search '''
    world.neighb_search()
    ''' [TIME END] neighb_search '''

    ''' sph pre-computation '''
    ''' [TIME START] DFSPH Part 1 '''
    world.step_sph_compute_compression_ratio()
    world.step_sph_compute_density()
    world.step_df_compute_beta()
    # world.step_sph_compute_density()
    # world.step_df_compute_alpha()
    ''' [TIME START] DFSPH Part 1 '''

    ''' pressure accleration (divergence-free) '''
    ''' [TIME START] DFSPH Part 2 '''
    world.step_vfsph_div(update_vel=True)
    # world.step_df_div()
    ''' [TIME END] DFSPH Part 2 '''
    print('div_free iter:', fluid_part.m_solver_df.div_free_iter[None])

    ''' [TIME START] Advection process '''
    world.clear_acc()
    fluid_part.getSolverAdv().add_acc_gravity()
    fluid_part.m_solver_sph.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.getSolverAdv().inloop_accumulate_vis_acc)
    fluid_part.m_solver_sph.loop_neighb(fluid_part.m_neighb_search.neighb_pool, bound_part, fluid_part.getSolverAdv().inloop_accumulate_vis_acc)
    world.acc2vel()
    ''' [TIME END] Advection process '''

    ''' pressure accleration (divergence-free) '''
    '''  [TIME START] DFSPH Part 3 '''
    world.step_vfsph_incomp(update_vel=True)
    # world.step_df_incomp()
    '''  [TIME START] DFSPH Part 3 '''
    print('incomp iter:', fluid_part.m_solver_df.incompressible_iter[None])

    ''' update particle position from velocity '''
    '''  [TIME START] DFSPH Part 4 '''
    world.update_pos_from_vel()
    '''  [TIME START] DFSPH Part 4 '''

    ''' cfl condition update'''
    '''  [TIME START] CFL '''
    # world.cfl_dt(0.4, max_time_step)
    '''  [TIME END] CFL '''

''' Viusalization and run '''
def vis_run(loop):
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
            sim_time += world.getWorldDt()
            print("loop ", loop_count)
            
            if(sim_time > timer*inv_fps):
                if gui.op_write_file:
                    sense_grid_part.copy_attr(from_attr=sense_grid_part.sph.density, to_attr=sense_grid_part.sensed_density)
                    sense_grid_part.m_solver_sph.sph_avg_velocity(sense_grid_part.m_neighb_search.neighb_pool)
                    sense_grid_part.m_solver_sph.sph_avg_strainRate(sense_grid_part.m_neighb_search.neighb_pool)
                    sense_output.export_to_numpy(index=timer,path=output_path, compressed=True)

                timer += 1
                flag_write_img = True
        if gui.op_refresh_window:
            gui.scene_setup()
            # gui.scene_add_parts_colorful(obj_pos=fluid_part.pos, obj_color=fluid_part.rgb,index_count=fluid_part.getObjStackTop(),size=world.g_part_size[None])
            gui.scene_add_parts(obj_pos=fluid_part.pos, obj_color=(0,0,1),index_count=fluid_part.getObjStackTop(),size=world.getPartSize())
            # gui.scene_add_parts(obj_pos=bound_part.pos, obj_color=(0,0.5,1),index_count=bound_part.getObjStackTop(),size=world.g_part_size[None])
            gui.canvas.scene(gui.scene)  # Render the scene

            if gui.op_save_img and flag_write_img:
                gui.window.save_image('output/'+str(timer)+'.png')
                flag_write_img = False

            gui.window.show()
        
        if timer > output_frame_num:
            break

def run(loop):
    inv_fps = 1/fps
    timer = 0
    sim_time = 0
    loop_count = 0

    while timer<output_frame_num:
        loop()
        loop_count += 1
        sim_time += world.getWorldDt()
        if(sim_time > timer*inv_fps):
            sense_grid_part.copy_attr(from_attr=sense_grid_part.sph.density, to_attr=sense_grid_part.sensed_density)
            sense_grid_part.m_solver_sph.sph_avg_velocity(sense_grid_part.m_neighb_search.neighb_pool)
            sense_grid_part.m_solver_sph.sph_avg_strainRate(sense_grid_part.m_neighb_search.neighb_pool)
            sense_output.export_to_numpy(index=timer,path=output_path, compressed=True)

            timer += 1


''' RUN THE SIMULATION '''
prep()
run(loop_ism)







