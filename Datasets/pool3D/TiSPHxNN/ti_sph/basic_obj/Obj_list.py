import taichi as ti

@ti.data_oriented
class Obj_list:
    def __init__(
        self,
    ):
        self.total_obj_num = 0
        self.obj = []
        self.obj_id_pair = {}
    
    def add_obj(self, obj):
        if (obj in self.obj):
            raise Exception("obj already in list")
        self.obj.append(obj)
        self.obj_id_pair[obj] = self.total_obj_num
        obj.id[None] = self.total_obj_num
        self.total_obj_num += 1
    
    def get_obj(self, obj_id):
        return self.obj[obj_id]
    
    def get_obj_id(self, obj):
        return obj.id[None]