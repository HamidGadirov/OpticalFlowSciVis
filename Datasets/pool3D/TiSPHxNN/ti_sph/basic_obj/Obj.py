import taichi as ti
from ..basic_op.type import *

@ti.data_oriented
class Obj:
    def __init__(self, is_dynamic: bool = True) -> None:
        self.m_is_dynamic = is_dynamic

    # def setObjId(self, id):
    #     self.m_id = val_i(id)

    # def setObjWorld(self, world):
    #     self.m_world = world

    # def getObjId(self):
    #     return self.m_id

    # @ti.func
    # def ti_getObjId(self):
    #     return self.m_id