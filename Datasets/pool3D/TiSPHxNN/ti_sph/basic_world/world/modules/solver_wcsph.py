import taichi as ti
import numpy as np
from typing import List
from ....basic_obj.Obj_Particle import Particle

from ....basic_solvers.Solver_wcsph import WCSPH_solver

def init_solver_wcsph(self):
    self.wcsph_solver_list:List[WCSPH_solver] = []
    for part_obj in self.part_obj_list:
        if part_obj.m_solver_wcsph is not None:
            self.wcsph_solver_list.append(part_obj)
    
def step_wcsph_add_acc_pressure(self):
    for part_obj in self.wcsph_solver_list:
        part_obj.m_solver_wcsph.compute_B()
        part_obj.m_solver_wcsph.ReLU_density()
        part_obj.m_solver_wcsph.compute_pressure()
    
    for part_obj in self.wcsph_solver_list:
        if part_obj.m_is_dynamic is True:
            for neighb_obj in part_obj.m_neighb_search.neighb_pool.neighb_obj_list:
                part_obj.m_solver_wcsph.loop_neighb(part_obj.m_neighb_search.neighb_pool, neighb_obj, part_obj.m_solver_wcsph.inloop_add_acc_pressure)

def step_wcsph_add_acc_number_density_pressure(self):
    for part_obj in self.wcsph_solver_list:
        part_obj.m_solver_wcsph.compute_B()
        part_obj.m_solver_wcsph.ReLU_density()
        part_obj.m_solver_wcsph.compute_pressure()
    
    for part_obj in self.wcsph_solver_list:
        if part_obj.m_is_dynamic is True:
            for neighb_obj in part_obj.m_neighb_search.neighb_pool.neighb_obj_list:
                part_obj.m_solver_wcsph.loop_neighb(part_obj.m_neighb_search.neighb_pool, neighb_obj, part_obj.m_solver_wcsph.inloop_add_acc_number_density_pressure)