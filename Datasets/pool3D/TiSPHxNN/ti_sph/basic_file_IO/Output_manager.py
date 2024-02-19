import taichi as ti
import numpy as np

from typing import List
from enum import Enum   

from ..basic_obj.Obj import Obj

@ti.data_oriented
class Output_manager:

    class type(Enum):
        SEQ  = 0 # Organize all data in a sequential way.
        GRID = 1 # Organize all data in a grid way. This requires Obj to have {obj_name.node_index=ti.field(ti.f32, shape=(grid_node_num, dim))}.

    def __init__(self, format_type:type, data_source:Obj) -> None:
        if not isinstance(format_type, self.type):
            raise ValueError(f"Invalid format type: {format_type}")
        if type is self.type.GRID and not hasattr(data_source, "node_index"):
            raise ValueError(f"Obj {data_source.__class__.__name__} does not have node_index field.")

        self.format_type = format_type
        self.obj = data_source

        self.data_name_list:List[str] = []
        self.type_list:List[str] = []
        self.one_time_data_name_list:List[str] = []
        self.one_time_type_list:List[str] = []

        # TODO: Consider 3D case.
        if self.format_type is self.type.GRID:
            self.np_node_index = self.obj.node_index.to_numpy()
            rows = self.np_node_index[:,0].max() + 1
            cols = self.np_node_index[:,1].max() + 1
            self.np_node_index_organized = np.empty((rows, cols, 2), dtype=int)
            self.np_data_organized = np.empty((rows, cols), dtype=float)

    def add_output_dataType(self, name:str, type:str = 'scalar'):
        if not hasattr(self.obj, name):
            raise ValueError(f"{name} is not in {self.obj.__class__.__name__}.")
        self.data_name_list.append(name)
        self.type_list.append(type)
    
    def add_one_time_output_dataType(self, name:str, type:str = 'scalar'):
        if not hasattr(self.obj, name):
            raise ValueError(f"{name} is not in {self.obj.__class__.__name__}.")
        self.one_time_data_name_list.append(name)
        self.one_time_type_list.append(type)
            
    
    def export_to_numpy(self, index:int=0, path:str=".", compressed=False):
        
        for data_name in self.data_name_list:

            file_name = f"{path}/{data_name}_{index}"
            np_data = getattr(self.obj, data_name).to_numpy()

            if self.format_type is self.type.SEQ:
                if compressed:
                    np.savez_compressed(file_name, np_data)
                else:
                    np.save(file_name, np_data)
    
    def export_one_time_to_numpy(self, path:str=".", compressed=False):
        
        for data_name in self.one_time_data_name_list:

            file_name = f"{path}/{data_name}"
            np_data = getattr(self.obj, data_name).to_numpy()

            if self.format_type is self.type.SEQ:
                if compressed:
                    np.savez_compressed(file_name, np_data)
                else:
                    np.save(file_name, np_data)
