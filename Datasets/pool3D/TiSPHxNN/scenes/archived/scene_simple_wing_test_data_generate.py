import taichi as ti
from ti_sph import *
from template_part import part_template
from template_grid import grid_template
import time
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

''' TAICHI SETTINGS '''
# ti.init(arch=ti.cuda, kernel_profiler=True) 
ti.init(arch=ti.cuda, device_memory_GB=5) # Use GPU
# ti.init(arch=ti.cpu) # Use CPU

''' GLOBAL SETTINGS '''
fps = 240
output_frame_num = 5000
sense_res = 512
output_shift = 0

part_size = 0.01
max_time_step = part_size/100
world = World(dim=2, lb = -15, rt = 15)
world.setWorldPartSize(part_size)
world.setWorldDt(max_time_step)

# Speciliazed coefficents for this scene
mov_vel = vec2_f([15.0, 0.0])
kinematic_viscosity_air = val_f(1.57e-5)
world.g_gravity[None] = vec2f(0, -9.8) * 10

''' Object position '''
# HINT: span=world.g_part_size[None]*(1+43e-5) can be more accurate but slower
wing_data = Wing2412_data_2D_with_cube(span=world.g_part_size[None]*(1+43e-5), chord_length=2.0, pos=vec2f(0,0), 
                                       cube_lb=vec2f(-8, -2), cube_rt=vec2f(3, 2), attack_angle=30)
sense_cell_size = val_f(0.05/sense_res*64)
sense_cube_data = Cube_data(type=Cube_data.FIXED_GRID_RES, span=sense_cell_size[None], grid_res=vec2i(sense_res,sense_res),grid_center=vec2f(1.5,0))

'''BASIC SETTINGS FOR FLUID'''
fluid_rest_density = val_f(1000)
'''INIT AN FLUID PARTICLE OBJECT'''
fluid_part_num = val_i(wing_data.fluid_num)
print("fluid_part_num", fluid_part_num)
fluid_part = world.add_part_obj(part_num=fluid_part_num[None], size=world.g_part_size, is_dynamic=True)
fluid_part.instantiate_from_template(part_template)
'''PUSH PARTICLES TO THE OBJECT'''
fluid_part.open_stack(val_i(wing_data.fluid_num))
fluid_part.fill_open_stack_with_nparr(fluid_part.pos, wing_data.fluid_pos)
fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.getObjPartSize())
fluid_part.fill_open_stack_with_val(fluid_part.volume, val_f(fluid_part.getObjPartSize()**world.g_dim[None]))
fluid_part.fill_open_stack_with_val(fluid_part.mass, val_f(fluid_rest_density[None]*fluid_part.getObjPartSize()**world.g_dim[None]))
fluid_part.fill_open_stack_with_val(fluid_part.rest_density, fluid_rest_density)
fluid_part.fill_open_stack_with_val(fluid_part.rgb, vec3_f([0.0, 0.0, 1.0]))
fluid_part.fill_open_stack_with_val(fluid_part.k_vis, kinematic_viscosity_air)
fluid_part.fill_open_stack_with_val(fluid_part.vel, mov_vel)
fluid_part.close_stack()


''' INIT BOUNDARY PARTICLE OBJECT '''
bound_rest_density = val_f(1000)
wing_part = world.add_part_obj(part_num=wing_data.wing_num, size=world.g_part_size, is_dynamic=False)
wing_part.instantiate_from_template(part_template)
wing_part.open_stack(val_i(wing_data.wing_num))
wing_part.fill_open_stack_with_nparr(wing_part.pos, wing_data.wing_pos)
wing_part.fill_open_stack_with_val(wing_part.size, wing_part.getObjPartSize())
wing_part.fill_open_stack_with_val(wing_part.volume, val_f(wing_part.getObjPartSize()**world.g_dim[None]))
wing_part.fill_open_stack_with_val(wing_part.mass, val_f(bound_rest_density[None]*wing_part.getObjPartSize()**world.g_dim[None]))
wing_part.fill_open_stack_with_val(wing_part.rest_density, bound_rest_density)
wing_part.fill_open_stack_with_val(wing_part.k_vis, kinematic_viscosity_air)
# wing_part.fill_open_stack_with_val(wing_part.vel, vec2_f([-0.5, 0.0]))
# wing_part.fill_open_stack_with_val(wing_part.vel_adv, vec2_f([-0.5, 0.0]))
# wing_part.fill_open_stack_with_val(wing_part.sph_df.vel_adv, vec2_f([-0.5, 0.0]))
wing_part.close_stack()

bound_part = world.add_part_obj(part_num=wing_data.bound_num, size=world.g_part_size, is_dynamic=False)
bound_part.instantiate_from_template(part_template)
bound_part.open_stack(val_i(wing_data.bound_num))
bound_part.fill_open_stack_with_nparr(bound_part.pos, wing_data.bound_pos)
bound_part.fill_open_stack_with_val(bound_part.size, bound_part.getObjPartSize())
bound_part.fill_open_stack_with_val(bound_part.volume, val_f(bound_part.getObjPartSize()**world.g_dim[None]))
bound_part.fill_open_stack_with_val(bound_part.mass, val_f(bound_rest_density[None]*bound_part.getObjPartSize()**world.g_dim[None]))
bound_part.fill_open_stack_with_val(bound_part.rest_density, bound_rest_density)
bound_part.fill_open_stack_with_val(bound_part.k_vis, kinematic_viscosity_air)
# bound_part.fill_open_stack_with_val(bound_part.acc, vec2_f([1.0, 0.0]))
# bound_part.fill_open_stack_with_val(bound_part.vel, vec2_f([1.0, 0.0]))
# bound_part.fill_open_stack_with_val(bound_part.vel_adv, vec2_f([1.0, 0.0]))
bound_part.close_stack()
bound_part.set(bound_part.vel, mov_vel)
bound_part.copy_attr(from_attr=bound_part.vel, to_attr=bound_part.vel_adv)
bound_part.copy_attr(from_attr=bound_part.vel, to_attr=bound_part.sph_df.vel_adv)

sense_grid_part = world.add_part_obj(part_num=sense_cube_data.num, size=sense_cell_size, is_dynamic=False)
sense_grid_part.instantiate_from_template(grid_template)
sense_grid_part.open_stack(val_i(sense_cube_data.num))
sense_grid_part.fill_open_stack_with_nparr(sense_grid_part.pos, sense_cube_data.pos)
sense_grid_part.fill_open_stack_with_nparr(sense_grid_part.node_index, sense_cube_data.index)
sense_grid_part.fill_open_stack_with_val(sense_grid_part.size, sense_grid_part.getObjPartSize())
sense_grid_part.fill_open_stack_with_val(sense_grid_part.volume, val_f(sense_grid_part.getObjPartSize()**world.g_dim[None]))
sense_grid_part.close_stack()

'''INIT NEIGHBOR SEARCH OBJECTS'''
neighb_list=[
    fluid_part, 
    wing_part,
    bound_part,
             ]


fluid_part.add_module_neighb_search()
wing_part.add_module_neighb_search()
bound_part.add_module_neighb_search()
sense_grid_part.add_module_neighb_search(max_neighb_num=val_i(fluid_part.getObjPartNum()*32))

fluid_part.add_neighb_objs(neighb_list)
wing_part.add_neighb_objs(neighb_list)
bound_part.add_neighb_objs(neighb_list)
sense_grid_part.add_neighb_obj(neighb_obj=fluid_part, search_range=val_f(sense_cell_size[None]*2))
# sense_grid_part.add_neighb_obj(neighb_obj=wing_part, search_range=val_f(sense_cell_size[None]*2))


fluid_part.add_solver_adv()
fluid_part.add_solver_sph()
fluid_part.add_solver_df(div_free_threshold=2e-4, incompressible_iter_max=200)

wing_part.add_solver_sph()
wing_part.add_solver_df(div_free_threshold=2e-4, incompressible_iter_max=200)

bound_part.add_solver_sph()
bound_part.add_solver_df(div_free_threshold=2e-4, incompressible_iter_max=200)

sense_grid_part.add_solver_sph()

world.init_modules()

world.neighb_search()

sense_output = Output_manager(format_type = Output_manager.type.SEQ, data_source = sense_grid_part)
sense_output.add_output_dataType("pos",2)
sense_output.add_output_dataType("node_index",2)
sense_output.add_output_dataType("sensed_density",1)
sense_output.add_output_dataType("vel",2)

# print('DEBUG sense_output', sense_output.np_node_index_organized)
# save as numpy file
# np.save("pos_np.npy", sense_output.np_node_index_organized)

def loop():
    world.update_pos_in_neighb_search()

    world.neighb_search()
    world.step_sph_compute_density()
    world.step_df_compute_alpha()
    world.step_df_div()
    print('div_free iter:', fluid_part.m_solver_df.div_free_iter[None])

    fluid_part.color_code_part(fluid_part.vel, 0, mov_vel[None][0]*1.5)

    # Sense grid operation
    sense_grid_part.clamp_val_to_arr(sense_grid_part.sph.density, 0, fluid_rest_density[None], sense_grid_part.rgb)
    sense_grid_part.copy_attr(from_attr=sense_grid_part.sph.density, to_attr=sense_grid_part.sensed_density)
    sense_grid_part.m_solver_sph.sph_avg_velocity(sense_grid_part.m_neighb_search.neighb_pool)

    # Advection
    world.clear_acc()
    world.add_acc_gravity()
    fluid_part.m_solver_sph.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_adv.inloop_accumulate_vis_acc)
    fluid_part.m_solver_sph.loop_neighb(fluid_part.m_neighb_search.neighb_pool, bound_part, fluid_part.m_solver_adv.inloop_accumulate_vis_acc)
    fluid_part.m_solver_sph.loop_neighb(fluid_part.m_neighb_search.neighb_pool, wing_part, fluid_part.m_solver_adv.inloop_accumulate_vis_acc)
    world.acc2vel_adv()
    print(bound_part.vel[0], bound_part.vel_adv[0], bound_part.sph_df.vel_adv[0])
    
    world.step_df_incomp()
    print('incomp iter:', fluid_part.m_solver_df.incompressible_iter[None])

    world.update_pos_from_vel()
    # wing_part.integ(wing_part.vel, world.g_dt[None], wing_part.pos)
    
    bound_part.integ(bound_part.vel, world.g_dt[None], bound_part.pos)
    world.cfl_dt(0.5, max_time_step)




def run(loop):
    inv_fps = 1/fps
    timer = 0
    sim_time = 0
    loop_count = 0

    while True:
        loop()
        loop_count += 1
        sim_time += world.g_dt[None]
        if(sim_time > timer*inv_fps):
            sense_output.export_to_numpy(index=output_shift+timer,path='./output')
            timer += 1
        if timer > output_frame_num:
            break

def vis_run(loop):
    inv_fps = 1/fps
    timer = 0
    sim_time = 0
    loop_count = 0

    gui = Gui3d()
    save_img=False
    while gui.window.running:

        gui.monitor_listen()

        if gui.op_system_run:
            loop()
            loop_count += 1
            print(loop_count)
            sim_time += world.g_dt[None]
            # print('loop count', loop_count, 'compressible ratio', 'incompressible iter', fluid_part_1.m_solver_df.incompressible_iter[None], ' ', fluid_part_2.m_solver_df.incompressible_iter[None])
            # print('comp ratio', fluid_part_1.m_solver_df.compressible_ratio[None], ' ', fluid_part_2.m_solver_df.compressible_ratio[None])
            # print('dt', world.g_dt[None])
            if(sim_time > timer*inv_fps):
                if gui.op_write_file:
                    sense_output.export_to_numpy(index=output_shift+timer,path='./output')
                timer += 1
                save_img = True
        
        if gui.op_refresh_window:
            gui.scene_setup()
            if gui.show_bound:
                gui.scene_add_parts_colorful(obj_pos=fluid_part.pos, obj_color=fluid_part.rgb,index_count=fluid_part.getObjStackTop(),size=world.g_part_size[None])
                gui.scene_add_parts(obj_pos=wing_part.pos, obj_color=(1,0.5,0),index_count=wing_part.getObjStackTop(),size=world.g_part_size[None])
                gui.scene_add_parts(obj_pos=bound_part.pos, obj_color=(0,1,1),index_count=bound_part.getObjStackTop(),size=world.g_part_size[None])
            else:
                gui.scene_add_parts_colorful(obj_pos=sense_grid_part.pos, obj_color=sense_grid_part.rgb, index_count=sense_grid_part.getObjStackTop(), size=sense_grid_part.getObjPartSize()*1.0)
            
            gui.canvas.scene(gui.scene)  # Render the scene

            if gui.op_save_img and save_img:
                gui.window.save_image('output/'+str(timer)+'.png')
                save_img = False

            gui.window.show()
        
        if timer > output_frame_num:
            break

loop()
vis_run(loop)







