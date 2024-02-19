import taichi as ti
import numpy as np
from .Data_generator import *

@ti.data_oriented
class Ply_data(Data_generator):
    def __init__(self, file_path: str):

        self.file_path = file_path

        self.pos = self.extract_vertex_data_from_ply()
        self.num = self.pos.shape[0]

    def extract_vertex_data_from_ply(self):
        # List to store vertices
        vertices = []
        
        with open(self.file_path, "r") as file:
            lines = file.readlines()
            # Finding the index of the "end_header" to start reading vertex data
            start_index = lines.index("end_header\n") + 1
            
            # Looping through the vertex data
            for line in lines[start_index:]:
                # If the line doesn't contain vertex data, break
                if len(line.split()) != 3:
                    break
                x, y, z = map(float, line.split())
                vertices.append([x, y, z])
        
        # Converting list of vertices to a NumPy array
        return np.array(vertices)