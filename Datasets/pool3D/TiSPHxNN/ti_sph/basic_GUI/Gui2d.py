from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import taichi as ti
from ..basic_op.type import *

class Gui2d:
    def __init__(self, objs, radius:ti.f32, lb:ti.types.vector(2, ti.f32), rt:ti.types.vector(2, ti.f32), dpi=200):

        self.objs = objs
        self.radius = radius

        self.lb = lb
        self.rt = rt
        self.dpi = dpi
    
    def save_img(self, path):
        plt.figure()

        plt.xlim(self.lb[0], self.rt[0])
        plt.ylim(self.lb[1], self.rt[1])

        plt.xlabel('X-axis (m)')
        plt.ylabel('Y-axis (m)')

        plt.axis('square')

        plt.title('2D Position Plot with RGB Colors')
        # plt.grid(True)

        for obj in self.objs:
            positoins = obj.getPosArr().to_numpy()
            colors = obj.rgb.to_numpy()

            x = positoins[:, 0]
            y = positoins[:, 1]

            plt.scatter(x, y, s=self.radius*10, c=colors, edgecolors='none') 
        
        ax = plt.gca()
        ax.set_facecolor('white')
        fig = plt.gcf()
        fig.set_facecolor('white')

        plt.savefig(path, dpi=self.dpi)
        plt.close()