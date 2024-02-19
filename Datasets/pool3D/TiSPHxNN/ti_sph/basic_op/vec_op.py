import numpy as np
import taichi as ti

@ti.func
def assign(x: ti.template(), y: ti.template()):
    for i in ti.static(range(x.n)):
        x[i] = y[i]
