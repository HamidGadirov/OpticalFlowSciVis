import taichi as ti

def init_solver_adv(self):
    self.adv_solver_list = []
    for part_obj in self.part_obj_list:
        if (part_obj.m_solver_adv is not None):
            self.adv_solver_list.append(part_obj)

def clear_acc(self):
    for part_obj in self.adv_solver_list:
        part_obj.m_solver_adv.clear_acc()

def add_acc_gravity(self):
    for part_obj in self.adv_solver_list:
        part_obj.m_solver_adv.add_acc_gravity()

def acc2vel_adv(self):
    for part_obj in self.adv_solver_list:
        part_obj.m_solver_adv.acc2vel_adv()

def acc2vel(self):
    for part_obj in self.adv_solver_list:
        part_obj.m_solver_adv.acc2vel()

def vel_adv2vel(self):
    for part_obj in self.adv_solver_list:
        part_obj.m_solver_adv.vel_adv2vel()

def update_pos_from_vel(self):
    for part_obj in self.adv_solver_list:
        part_obj.m_solver_adv.update_pos()
