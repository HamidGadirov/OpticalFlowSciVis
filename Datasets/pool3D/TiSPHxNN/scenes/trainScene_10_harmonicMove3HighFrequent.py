import taichi as ti
from ti_sph import *
from template_part import part_template
from template_grid import grid_template
import time
import sys
import numpy as np
from enum import Enum
np.set_printoptions(threshold=sys.maxsize)

class mode(Enum):
    DEBUG = 0
    RELEASE = 1
MODE = mode.RELEASE
dpi = 200

''' TAICHI SETTINGS '''
if(MODE == mode.DEBUG):
    ti.init(arch=ti.cuda, debug=True, device_memory_GB=6)
elif(MODE == mode.RELEASE):
    ti.init(arch=ti.cuda, device_memory_GB=22) 

''' GLOBAL SETTINGS '''
output_shift = 0
output_frame_num = 2000
fps = 30
sense_res = 258

sense_cell_size = val_f(7.0/sense_res)
part_size = sense_cell_size[None] / 10
if(MODE == mode.DEBUG): part_size *= 24

max_time_step = part_size/100
k_vis = 5e-5
world = World(dim=2)
world.setWorldPartSize(part_size)
world.setWorldDt(max_time_step)
world.setWorldObjNum(5)

print('default dt', world.getWorldDt())

'''BASIC SETTINGS FOR objects'''
span = world.getPartSize()*1.001

fluid_cube_data_1 = Cube_data(type=Cube_data.FIXED_CELL_SIZE, lb=vec2f(-4+span, -4+span), rt=vec2f(4-span*3, 3), span=span)
fluid_part_num = fluid_cube_data_1.num

box_data   = Box_data(lb=vec2f(-6, -4), rt=vec2f(4, 4), span=world.getPartSize()*1.001, layers=3)
rod_data_1 = Box_data(lb=vec2f(-5.5,  1.5), rt=vec2f(-5.0,  3.0), span=world.getPartSize()*1.001, layers=3)
rod_data_2 = Box_data(lb=vec2f(-5.5, -1.0), rt=vec2f(-5.0,  0.5), span=world.getPartSize()*1.001, layers=3)
rod_data_3 = Box_data(lb=vec2f(-5.5, -3.5), rt=vec2f(-5.0, -2.0), span=world.getPartSize()*1.001, layers=3)
# rod_data = Box_data(lb=vec2f(-5-span*2, -4+span*3), rt=vec2f(-5+span*2, 4-span*4), span=world.getPartSize()*1.001, layers=3)

'''INIT AN FLUID PARTICLE OBJECT'''
fluid_rest_density = 1000
print("fluid_part_num", fluid_part_num)
fluid_part = Particle(part_num=fluid_part_num, part_size=world.g_part_size, is_dynamic=True)
world.attachPartObj(fluid_part)
fluid_part.instantiate_from_template(part_template, world)

'''PUSH PARTICLES TO THE OBJECT'''
fluid_part.open_stack(fluid_cube_data_1.num)
fluid_part.fill_open_stack_with_nparr(fluid_part.pos, fluid_cube_data_1.pos)
fluid_part.fill_open_stack_with_val(fluid_part.size, fluid_part.getObjPartSize())
fluid_part.fill_open_stack_with_val(fluid_part.volume, fluid_part.getObjPartSize()**world.getWorldDim())
fluid_part.fill_open_stack_with_val(fluid_part.mass, fluid_rest_density*fluid_part.getObjPartSize()**world.getWorldDim())
fluid_part.fill_open_stack_with_val(fluid_part.rest_density, fluid_rest_density)
fluid_part.fill_open_stack_with_val(fluid_part.rgb, vec3f([0.0, 0.0, 1.0]))
fluid_part.fill_open_stack_with_val(fluid_part.k_vis, k_vis)
fluid_part.close_stack()

''' INIT BOUNDARY PARTICLE OBJECT '''
bound_rest_density = 1000
bound_part = Particle(part_num=box_data.num, part_size=world.g_part_size, is_dynamic=False)
world.attachPartObj(bound_part)
bound_part.instantiate_from_template(part_template, world)
bound_part.open_stack(box_data.num)
bound_part.fill_open_stack_with_arr(bound_part.pos, box_data.pos)
bound_part.fill_open_stack_with_val(bound_part.size, bound_part.getObjPartSize())
bound_part.fill_open_stack_with_val(bound_part.volume, bound_part.getObjPartSize()**world.getWorldDim())
bound_part.fill_open_stack_with_val(bound_part.mass, bound_rest_density*bound_part.getObjPartSize()**world.getWorldDim())
bound_part.fill_open_stack_with_val(bound_part.rest_density, bound_rest_density)
fluid_part.fill_open_stack_with_val(bound_part.rgb, vec3f([0.0, 0.0, 0.0]))
bound_part.fill_open_stack_with_val(bound_part.k_vis, k_vis)
bound_part.close_stack()


''' INIT ROD OBJECT '''
rod_rest_density = 1000
rod_part_1 = Particle(part_num=rod_data_1.num, part_size=world.g_part_size, is_dynamic=False)
world.attachPartObj(rod_part_1)
rod_part_1.instantiate_from_template(part_template, world)
rod_part_1.open_stack(rod_data_1.num)
rod_part_1.fill_open_stack_with_arr(rod_part_1.pos, rod_data_1.pos)
rod_part_1.fill_open_stack_with_val(rod_part_1.size, rod_part_1.getObjPartSize())
rod_part_1.fill_open_stack_with_val(rod_part_1.volume, rod_part_1.getObjPartSize()**world.getWorldDim())
rod_part_1.fill_open_stack_with_val(rod_part_1.mass, rod_rest_density*rod_part_1.getObjPartSize()**world.getWorldDim())
rod_part_1.fill_open_stack_with_val(rod_part_1.rest_density, rod_rest_density)
rod_part_1.fill_open_stack_with_val(rod_part_1.rgb, vec3f([0.0, 0.0, 0.0]))
rod_part_1.fill_open_stack_with_val(rod_part_1.k_vis, k_vis)
rod_part_1.close_stack()

rod_part_2 = Particle(part_num=rod_data_2.num, part_size=world.g_part_size, is_dynamic=False)
world.attachPartObj(rod_part_2)
rod_part_2.instantiate_from_template(part_template, world)
rod_part_2.open_stack(rod_data_2.num)
rod_part_2.fill_open_stack_with_arr(rod_part_2.pos, rod_data_2.pos)
rod_part_2.fill_open_stack_with_val(rod_part_2.size, rod_part_2.getObjPartSize())
rod_part_2.fill_open_stack_with_val(rod_part_2.volume, rod_part_2.getObjPartSize()**world.getWorldDim())
rod_part_2.fill_open_stack_with_val(rod_part_2.mass, rod_rest_density*rod_part_2.getObjPartSize()**world.getWorldDim())
rod_part_2.fill_open_stack_with_val(rod_part_2.rest_density, rod_rest_density)
rod_part_2.fill_open_stack_with_val(rod_part_2.rgb, vec3f([0.0, 0.0, 0.0]))
rod_part_2.fill_open_stack_with_val(rod_part_2.k_vis, k_vis)
rod_part_2.close_stack()

rod_part_3 = Particle(part_num=rod_data_3.num, part_size=world.g_part_size, is_dynamic=False)
world.attachPartObj(rod_part_3)
rod_part_3.instantiate_from_template(part_template, world)
rod_part_3.open_stack(rod_data_3.num)
rod_part_3.fill_open_stack_with_arr(rod_part_3.pos, rod_data_3.pos)
rod_part_3.fill_open_stack_with_val(rod_part_3.size, rod_part_3.getObjPartSize())
rod_part_3.fill_open_stack_with_val(rod_part_3.volume, rod_part_3.getObjPartSize()**world.getWorldDim())
rod_part_3.fill_open_stack_with_val(rod_part_3.mass, rod_rest_density*rod_part_3.getObjPartSize()**world.getWorldDim())
rod_part_3.fill_open_stack_with_val(rod_part_3.rest_density, rod_rest_density)
rod_part_3.fill_open_stack_with_val(rod_part_3.rgb, vec3f([0.0, 0.0, 0.0]))
rod_part_3.fill_open_stack_with_val(rod_part_3.k_vis, k_vis)
rod_part_3.close_stack()

sense_cube_data = Cube_data(type=Cube_data.FIXED_GRID_RES, span=sense_cell_size[None], grid_res=vec2i(sense_res,sense_res),grid_center=vec2f(0,0))
sense_grid_part = Particle(part_num=sense_cube_data.num, part_size=sense_cell_size, is_dynamic=False)
world.attachPartObj(sense_grid_part)
sense_grid_part.instantiate_from_template(grid_template, world)
sense_grid_part.open_stack(sense_cube_data.num)
sense_grid_part.fill_open_stack_with_nparr(sense_grid_part.pos, sense_cube_data.pos)
sense_grid_part.fill_open_stack_with_nparr(sense_grid_part.node_index, sense_cube_data.index)
sense_grid_part.fill_open_stack_with_val(sense_grid_part.size, sense_grid_part.getObjPartSize())
sense_grid_part.fill_open_stack_with_val(sense_grid_part.volume, sense_grid_part.getObjPartSize()**world.getWorldDim())
sense_grid_part.close_stack()

'''INIT NEIGHBOR SEARCH OBJECTS'''
neighb_list=[fluid_part, 
             bound_part, 
             rod_part_1, 
             rod_part_2,
             rod_part_3
             ]


fluid_part.add_module_neighb_search()
bound_part.add_module_neighb_search()
rod_part_1.add_module_neighb_search()
rod_part_2.add_module_neighb_search()
rod_part_3.add_module_neighb_search()
sense_grid_part.add_module_neighb_search(max_neighb_num=val_i(fluid_part.getObjPartNum()*32))

fluid_part.add_neighb_objs(neighb_list)
bound_part.add_neighb_objs(neighb_list)
rod_part_1.add_neighb_objs(neighb_list)
rod_part_2.add_neighb_objs(neighb_list)
rod_part_3.add_neighb_objs(neighb_list)
sense_grid_part.add_neighb_obj(neighb_obj=fluid_part, search_range=val_f(sense_cell_size[None]*2))


fluid_part.add_solver_adv()
fluid_part.add_solver_sph()
fluid_part.add_solver_df(div_free_threshold=1e-4)

bound_part.add_solver_sph()
bound_part.add_solver_df(div_free_threshold=1e-4)

rod_part_1.add_solver_sph()
rod_part_1.add_solver_adv()
rod_part_1.add_solver_df(div_free_threshold=1e-4)

rod_part_2.add_solver_sph()
rod_part_2.add_solver_adv()
rod_part_2.add_solver_df(div_free_threshold=1e-4)

rod_part_3.add_solver_sph()
rod_part_3.add_solver_adv()
rod_part_3.add_solver_df(div_free_threshold=1e-4)

sense_grid_part.add_solver_sph()

world.init_modules()

world.neighb_search()

sense_output = Output_manager(format_type = Output_manager.type.SEQ, data_source = sense_grid_part)
sense_output.add_output_dataType("pos")
sense_output.add_output_dataType("node_index")
sense_output.add_output_dataType("sensed_density")
sense_output.add_output_dataType("vel")
sense_output.add_output_dataType("strainRate")

# print('DEBUG sense_output', sense_output.np_node_index_organized)
# save as numpy file
# np.save("pos_np.npy", sense_output.np_node_index_organized)

def loop(is_log_loop=False):
    world.cfl_dt(0.5, max_time_step)

    world.update_pos_in_neighb_search()

    world.neighb_search()
    world.step_sph_compute_density()
    world.step_df_compute_alpha()
    world.step_df_div()
    # print('div_free iter:', fluid_part.m_solver_df.div_free_iter[None])
    
    if(is_log_loop):
        fluid_part.m_solver_sph.sph_compute_strainRate(fluid_part, fluid_part.m_neighb_search.neighb_pool)

    world.clear_acc()
    fluid_part.getSolverAdv().add_acc_gravity()
    rod_part_1.getSolverAdv().add_acc_harmonic(axis=0, dir = -1, period=1.5, amplitude=0.3, world_time=world.getTime(), delay=1)
    rod_part_2.getSolverAdv().add_acc_harmonic(axis=0, dir = -1, period=1.5, amplitude=0.3, world_time=world.getTime(), delay=0.5)
    rod_part_3.getSolverAdv().add_acc_harmonic(axis=0, dir = -1, period=1.5, amplitude=0.3, world_time=world.getTime(), delay=0)
    fluid_part.m_solver_sph.loop_neighb(fluid_part.m_neighb_search.neighb_pool, fluid_part, fluid_part.m_solver_adv.inloop_accumulate_vis_acc)
    fluid_part.m_solver_sph.loop_neighb(fluid_part.m_neighb_search.neighb_pool, bound_part, fluid_part.m_solver_adv.inloop_accumulate_vis_acc)
    world.acc2vel()
    
    world.step_df_incomp()
    # print('incomp iter:', fluid_part.m_solver_df.incompressible_iter[None])

    world.update_pos_from_vel()

    # print(' ')


def run(loop):
    inv_fps = 1/fps
    timer = int(0)
    sim_time = float(0.0)
    loop_count = int(0)

    gui2d_part = Gui2d(objs=[fluid_part, bound_part, rod_part_1, rod_part_2, rod_part_3], radius=world.getWorldPartSize(), lb=vec2f([-8,-8]),rt=vec2f([8,8]),dpi=dpi)
    gui2d_grid = Gui2d(objs=[sense_grid_part], radius=sense_grid_part.getObjPartSize(), lb=vec2f([-8,-8]),rt=vec2f([8,8]), dpi=dpi)

    while timer < output_frame_num:
        is_log_loop = sim_time > timer*inv_fps

        loop(is_log_loop)
        loop_count += 1
        sim_time += world.getWorldDt()
        world.setTime(sim_time)

        if(is_log_loop):
            
            sense_grid_part.copy_attr(from_attr=sense_grid_part.sph.density, to_attr=sense_grid_part.sensed_density)
            sense_grid_part.m_solver_sph.sph_avg_velocity(sense_grid_part.m_neighb_search.neighb_pool)
            
            sense_grid_part.m_solver_sph.sph_avg_strainRate(sense_grid_part.m_neighb_search.neighb_pool)

            sense_output.export_to_numpy(index=output_shift+timer,path='./output')

            gui2d_part.save_img(path='./output/part_'+str(output_shift+timer)+'.png')
            
            sense_grid_part.clamp_val_to_arr(sense_grid_part.sph.density, 0, 1000, sense_grid_part.rgb)
            gui2d_grid.save_img(path='./output/grid_'+str(output_shift+timer)+'.png')

            timer += 1

run(loop)







