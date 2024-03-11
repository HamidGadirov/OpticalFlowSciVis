# include simulation module
# PYSOLR_PATH = '/content/TiSPHxNN'
PYSOLR_PATH = './TiSPHxNN'
import sys
if not PYSOLR_PATH in sys.path:
    sys.path.append(PYSOLR_PATH)

#@title [Simulation] Run the simulation

import taichi as ti
# from ti_sph import *
import ti_sph as tsph
from template_part import part_template
from template_grid import grid_template
from Dataset_processing import *
import time
import sys
import numpy as np
import csv
np.set_printoptions(threshold=sys.maxsize)
import pyvista as pv
import imageio

# output_path = './output' #@param {type:"string"}
output_path = './output_1e4_64_3' #@param {type:"string"}


run_simulation = False # False True
if run_simulation:
    print("starting simulation...")
    ''' TAICHI SETTINGS '''
    # ti.init(arch=ti.cpu)
    # ti.init(arch=ti.metal)
    ti.init(arch=ti.cuda,device_memory_GB=20)
    # ti.init(arch=ti.cuda,device_memory_GB=10)

    ''' SETTINGS OUTPUT DATA '''
    # output fps
    fps = 30 #@param {type:"raw"} 30
    # max output frame number
    output_frame_num = 310 #@param {type:"raw"}

    ''' SETTINGS SIMULATION '''
    # scaling_factor = 2 # 64
    scaling_factor = 3.2 # 128
    # size of the particle
    part_size = 0.07
     # change to 0.06 faster + 0.05 default
    #  0.07 128
    # larger the particle larger the grid, smaller the simulation inside of the grid
    # 2931325 - part_size = 0.07 scaling_factor = 3.2

    # max time step size
    max_time_step = part_size / 100. # 50 - might make unstable
    # sense_cell_size = part_size * 2.5 # 2.5 64
    sense_cell_size = part_size * 1.4 # 128 1.4
    # 2.0 for 128 safe ratio
    # kinematic viscosity of fluid
    kinematic_viscosity_fluid = 1e-1
    
     # 1e-4 default
    # 1e-4 - water; 1e-2 1e-3 
    print("kinematic_viscosity_fluid", kinematic_viscosity_fluid)

    ''' INIT SIMULATION WORLD '''
    # create a 3D world
    # world = tsph.World(dim=3)
    world = tsph.World(dim=3, lb=-11, rt=11) # 128
    # set the particle diameter
    world.setWorldPartSize(part_size)
    # set the max time step size
    world.setWorldDt(max_time_step)

    ''' DATA SETTINGS FOR FLUID PARTICLE '''
    # generate the fluid particle data as a hollowed sphere, rotating irrotationally
    # x, y, z -> width, height, depth
    # fluid_cube_data = tsph.Cube_data(type=tsph.Cube_data.FIXED_CELL_SIZE, lb=tsph.vec3f(-0.5,-0.5,-0.5)*scaling_factor, rt=tsph.vec3f(0.5,1.5,0.5)*scaling_factor, span=world.getPartSize())
    # fluid_cube_data = tsph.Cube_data(type=tsph.Cube_data.FIXED_CELL_SIZE, lb=tsph.vec3f(-0.5,-0.5,-0.25)*scaling_factor, rt=tsph.vec3f(0.5,1.5,0.75)*scaling_factor, span=world.getPartSize())
    fluid_cube_data = tsph.Cube_data(type=tsph.Cube_data.FIXED_CELL_SIZE, lb=tsph.vec3f(-0.5,-0.5,0.)*scaling_factor, rt=tsph.vec3f(0.5,1.5,1.0)*scaling_factor, span=world.getPartSize())
    # fluid_cube_data = tsph.Cube_data(type=tsph.Cube_data.FIXED_CELL_SIZE, lb=tsph.vec3f(-0.5,-0.5,0.5)*scaling_factor, rt=tsph.vec3f(0.5,1.5,1.5)*scaling_factor, span=world.getPartSize())
    # change the cube shape as a hyperpapram
    pool_data = tsph.Squared_pool_3D_data(container_height=6*scaling_factor, container_size=4*scaling_factor, fluid_height=2*scaling_factor, span=world.getPartSize(), layer = 3)
    # sense_cube_data = tsph.Cube_data(type=tsph.Cube_data.FIXED_GRID_RES, span=sense_cell_size, grid_res=tsph.vec3i(64,64,64),grid_center=tsph.vec3f(0,-2,0))
    sense_cube_data = tsph.Cube_data(type=tsph.Cube_data.FIXED_GRID_RES, span=sense_cell_size, grid_res=tsph.vec3i(127,127,127),grid_center=tsph.vec3f(0,-2,0))
 
    # particle number of fluid/boundary
    fluid_part_num = pool_data.fluid_part_num + fluid_cube_data.num
    bound_part_num = pool_data.bound_part_num
    print("fluid_part_num", fluid_part_num)
    # position info of fluid/boundary (as numpy arrays)
    fluid_part_pos = pool_data.fluid_part_pos
    bound_part_pos = pool_data.bound_part_pos
    # initial velocity info of fluid

    print("------------------Information Statistics------------------")
    print(f"[BASIC INFO]particle size:" , part_size)
    print(f"[BASIC INFO]cell size: {sense_cell_size}")
    print(f"[BASIC INFO]ratio of cell size to particle size: {sense_cell_size/part_size}")
    print("")
    print(f"[WORLD INFO]left bottom corner of the world: {world.g_space_lb}")
    print(f"[WORLD INFO]right top corner of the world: {world.g_space_rt}")
    print("")
    print(f"[GRID INFO] grid resolution: {sense_cube_data.grid_res}")
    print(f"[GRID INFO] grid size = cell size * grid resolution = {sense_cell_size} * {sense_cube_data.grid_res} = ", sense_cell_size*sense_cube_data.grid_res)
    print(f"[GRID INFO] grid center: {(sense_cube_data.rt + sense_cube_data.lb) / 2} ")
    print(f"[GRID INFO] left bottom corner of the grid: {sense_cube_data.lb}")
    print(f"[GRID INFO] right top corner of the grid: {sense_cube_data.rt}")
    print("")
    print(f"[FLUID INFO] fluid particle number: {fluid_part_num}")
    print(f"[FLUID INFO] fluid cube size: {fluid_cube_data.rt - fluid_cube_data.lb}")
    print(f"[FLUID INFO] fluid cube left bottom corner: {fluid_cube_data.lb}")
    print(f"[FLUID INFO] fluid cube right top corner: {fluid_cube_data.rt}")
    print("")
    print(f"[POOL INFO] fluid pool left bottom corner: {tsph.vec3f(-pool_data.container_size/2, -pool_data.container_height/2, -pool_data.container_size/2)}")
    print(f"[POOL INFO] fluid pool right top corner: {tsph.vec3f(pool_data.container_size/2, pool_data.container_height/2, pool_data.container_size/2)}")
    print(f"[POOL INFO] fluid height position in the pool: {pool_data.fluid_height-pool_data.container_height/2}")

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
    # fluid_part.fill_open_stack_with_nparr(fluid_part.pos.to_numpy(), fluid_cube_data.pos)
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
                print("index = ", timer)


    ''' RUN THE SIMULATION '''
    prep()
    run(loop_ism)

else:
    print("starting creating organized output...")
    # end_index = 249 # 299
    end_index = 299
    # """
    #@title [Data Organization] Organize the output raw data to 3D shape

    ti.init(arch=ti.gpu)

    # input_path = './output' #@param {type:"string"}
    input_path = './output_1e4_64_3' #@param {type:"string"}
    # output_path = './organized_output' #@param {type:"string"}
    output_path = './organized_output_1e4_64_3' #@param {type:"string"}

    input_compressed = True
    output_compressed = True

    attr_list = ['vel', 'pos', 'sensed_density', 'strainRate']
    output_attr_list = ['velocity', 'position', 'density', 'strainRate']

    for attr, output_attr in zip(attr_list, output_attr_list):
        seqdata = tsph.SeqData(path=input_path, attr_name=attr, compressed=input_compressed)
        seqdata.end_index = end_index
        seqdata.reshape_to_3d(index_attr_name='node_index', output_path=output_path, output_name=output_attr, compressed=output_compressed)
        print("x")
    # """

    # """
    #@title [Visualization] Visualize the density and velocity field
    print("Visualization")

    pv.set_jupyter_backend('static')
    pv.global_theme.notebook = True
    pv.start_xvfb()

    # input_path = './organized_output'#@param {type:"string"}
    input_path = './organized_output_1e4_64_3'#@param {type:"string"}

    plotter_vel = pv.Plotter(off_screen=True)
    vel_animation = "animation_vel_1e4_64_3.gif" #@param {type:"string"}
    plotter_vel.open_gif(vel_animation)

    plotter_density = pv.Plotter(off_screen=True)
    density_animation = "animation_density_1e4_64_3.gif" #@param {type:"string"}
    plotter_density.open_gif(density_animation)

    start_index = 0 #@param {type:"raw"}
    # end_index = 160 # 300 #@param {type:"raw"}

    frames_vel = []
    frames_density = []

    for i in range(start_index, end_index+1):
        print(i)
        grid_density = pv.ImageData()

        field_density = np.load(f'{input_path}/density_{i}.npz')['arr_0']

        grid_density.dimensions = np.array(field_density.shape) + 1
        grid_density.spacing = (1, 1, 1)  # Assuming the spacing between your points is 1 in all directions
        grid_density.origin = (0, 0, 0)  # Assuming the origin is at (0, 0, 0)
        grid_density.cell_data["values"] = field_density.flatten(order="F")  # Flatten the array in Fortran order
        plotter_density.clear()
        plotter_density.add_volume(grid_density, scalars="values")
        outline = grid_density.outline_corners()
        plotter_density.add_mesh(outline, color="black", show_edges=True)
        plotter_density.camera_position = 'zy'
        plotter_density.camera.elevation = 30
        plotter_density.write_frame()

        grid_velocity = pv.ImageData()
        field_vel = np.load(f'{input_path}/velocity_{i}.npz')['arr_0']
        field_vel = np.sqrt(field_vel[...,0]**2+field_vel[...,1]**2+field_vel[...,2]**2)
        grid_velocity.dimensions = np.array(field_vel.shape) + 1
        grid_velocity.spacing = (1, 1, 1)  # Assuming the spacing between your points is 1 in all directions
        grid_velocity.origin = (0, 0, 0)  # Assuming the origin is at (0, 0, 0)
        grid_velocity.cell_data["values"] = field_vel.flatten(order="F")  # Flatten the array in Fortran order
        plotter_vel.clear()
        plotter_vel.add_volume(grid_velocity, scalars="values")
        outline = grid_velocity.outline_corners()
        plotter_vel.add_mesh(outline, color="black", show_edges=True)
        plotter_vel.camera_position = 'zy'
        plotter_vel.camera.elevation = 30
        plotter_vel.write_frame()

        # Append the rendered frames to the list
        frames_vel.append(plotter_vel.screenshot())
        frames_density.append(plotter_density.screenshot())

        # print("x")

    plotter_vel.close()
    plotter_density.close()

    # Save the animations explicitly
    imageio.mimsave(vel_animation, frames_vel, duration=0.1)
    imageio.mimsave(density_animation, frames_density, duration=0.1)
    # """