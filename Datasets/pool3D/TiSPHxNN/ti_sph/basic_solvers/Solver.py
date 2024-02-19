import taichi as ti
from ..basic_obj.Obj_Particle import Particle

ti.data_oriented
class Solver:
    def __init__(self, obj) -> None:
        self.obj = obj
    
    ''' get&set '''
    def getObj(self) -> Particle:
        return self.obj
    @ti.func
    def tiGetObj(self) -> Particle:
        return self.obj