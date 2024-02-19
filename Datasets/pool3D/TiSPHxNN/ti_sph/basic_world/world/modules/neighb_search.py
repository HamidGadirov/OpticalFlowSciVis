import taichi as ti

def init_neighb_search(self):
    self.neighb_search_list = []
    for part_obj in self.part_obj_list:
        if part_obj.m_neighb_search is not None:
            self.neighb_search_list.append(part_obj)

def update_pos_in_neighb_search(self):
    for part_obj in self.neighb_search_list:
        part_obj.m_neighb_search.update_self()

def update_neighbs(self):
    for part_obj in self.neighb_search_list:
        part_obj.m_neighb_search.search_neighbors()

def search_neighb(self):
    self.update_pos_in_neighb_search()
    self.update_neighbs()