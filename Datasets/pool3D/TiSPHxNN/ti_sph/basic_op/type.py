import taichi as ti

""" Type"""
vec2f = ti.types.vector(2, ti.f32)
vec2i = ti.types.vector(2, ti.i32)

vec3f = ti.types.vector(3, ti.f32)
vec3i = ti.types.vector(3, ti.i32)

vec4f = ti.types.vector(4, ti.f32)
vec4i = ti.types.vector(4, ti.i32)


def vecxf(n):
    return ti.types.vector(n, ti.f32)

def vecxm(n, m):
    return ti.types.matrix(n, m, ti.f32)

def vecxi(n):
    return ti.types.vector(n, ti.i32)

def vecx_i(n):
    return ti.Vector.field(n, ti.i32, ())

def vecx_ii(n):
    return ti.Vector.field(n, ti.i64, ())
    
def vecx_f(n):
    return ti.Vector.field(n, ti.f32, ())

def vec2_i(a=0):
    vec2i = ti.Vector.field(2, ti.i32, ())
    vec2i.fill(a)
    return vec2i

def vec2_f(a=0):
    vec2f = ti.Vector.field(2, ti.f32, ())
    vec2f.fill(a)
    return vec2f

def vec3_i(a=0):
    vec3i = ti.Vector.field(3, ti.i32, ())
    vec3i.fill(a)
    return vec3i

def vec3_f(a=0):
    vec3f = ti.Vector.field(3, ti.f32, ())
    vec3f.fill(a)
    return vec3f

def vec4_i(a=0):
    vec4i = ti.Vector.field(4, ti.i32, ())
    vec4i.fill(a)
    return vec4i

def vec4_f(a=0):
    vec4f = ti.Vector.field(4, ti.f32, ())
    vec4f.fill(a)
    return vec4f

def val_f(val=0):
    obj = ti.field(ti.f32, ())
    obj[None] = val
    return obj

def val_i(val=0):
    obj = ti.field(ti.i32, ())
    obj[None] = int(val)
    return obj

def val_ii(val=0):
    obj = ti.field(ti.i64, ())
    obj[None] = int(val)
    return obj