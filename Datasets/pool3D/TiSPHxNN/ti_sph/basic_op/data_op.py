import taichi as ti
from .type import *
""" Monocular """


@ti.kernel
def ker_arr_fill(
    to_arr: ti.template(),
    val: ti.template(),
    offset: ti.i32,
    num: ti.i32,
):
    for i in range(offset, offset + num):
        for n in ti.static(range(to_arr.n)):
            to_arr[i][n] = val


@ti.kernel
def ker_arr_set(
    to_arr: ti.template(),
    val: ti.template(),
    offset: ti.i32,
    num: ti.i32,
):
    for i in range(offset, offset + num):
        to_arr[i] = val


@ti.kernel
def ker_arr_add(
    to_arr: ti.template(),
    val: ti.template(),
    offset: ti.i32,
    num: ti.i32,
):
    for i in range(offset, offset + num):
        to_arr[i] += val

""" Binocular """


@ti.kernel
def ker_arr_cpy(
    to_arr: ti.template(),
    from_arr: ti.template(),
    offset: vec2i,  # offset[0] for to_arr, offset[1] for from_arr
    num: ti.i32,
):
    arr_n = ti.static(from_arr.n)
    for i in range(num):
        for n in ti.static(range(arr_n)):
            to_arr[i+offset[0]][n] = from_arr[i+offset[1]][n]

@ti.kernel
def ker_negative2zero_and_normalize(
    arr: ti.template()
):
    max_val = 0
    for I in ti.grouped(arr):
        if arr[I] < 0: arr[I] = 0 
        ti.atomic_max(max_val, arr[I])
    for I in ti.grouped(arr):
        arr[I] /= max_val

@ti.kernel
def ker_normalize(
    arr: ti.template()
):
    max_val = 0
    for I in ti.grouped(arr):
        ti.atomic_max(max_val, arr[I])
    for I in ti.grouped(arr):
        arr[I] /= max_val

@ti.kernel
def ker_binary(
    arr: ti.template(),
    threshold: ti.f32
):
    for I in ti.grouped(arr):
        if arr[I] > threshold: arr[I] = 1
        else: arr[I] = 0

@ti.kernel
def ker_invBinary(
    arr: ti.template(),
    threshold: ti.f32
):
    for I in ti.grouped(arr):
        if arr[I] > threshold: arr[I] = 0
        else: arr[I] = 1

@ti.kernel
def ker_binary_and_propagate(
    arr: ti.template(),
    threshold: ti.f32
):
    for I in ti.grouped(arr):
        if arr[I] > threshold: arr[I] = 1
        else: 
            for J in ti.grouped(ti.ndrange(5,5)):
                arr[I+J-2] = 0

@ti.kernel
def ker_entry_wise_product(
    in_arr_1: ti.template(),
    in_arr2: ti.template(),
    out_arr: ti.template(),
):
    for I in ti.grouped(in_arr_1):
        out_arr[I] = in_arr_1[I] * in_arr2[I]
    
@ti.kernel
def ker_entry_wise_productEqual(
    in_arr: ti.template(),
    out_arr: ti.template(),
):
    for I in ti.grouped(in_arr):
        out_arr[I] *= in_arr[I]

@ti.kernel
def ker_entry_wise_grad_mag(
    in_grad_x: ti.template(),
    in_grad_y: ti.template(),
    out_grad_mag: ti.template(),
):
    for I in ti.grouped(in_grad_x):
        out_grad_mag[I] = ti.sqrt(in_grad_x[I]**2 + in_grad_y[I]**2)
