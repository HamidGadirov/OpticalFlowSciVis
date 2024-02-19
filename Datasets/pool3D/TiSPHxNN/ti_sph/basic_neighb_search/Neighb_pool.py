import taichi as ti
from ..basic_op.type import *
from ..basic_solvers.sph_funcs import *
from ..basic_obj.Obj_Particle import Particle
from ..basic_solvers.Solver import Solver

'''#################### BELOW IS THE TEMPLATE FOR NEIGHBORHOOD SEASCHING ####################'''
@ti.data_oriented
class Neighb_search_template(Solver):
    def __init__(
        self,
        dim,  # int32
        search_cell_range,  # int32
    ):
        self.search_cell_range = val_i(0)
        self.search_cell_range[None] = search_cell_range

        self.search_template = ti.Vector.field(
            dim, ti.i32, (self.search_cell_range[None] * 2 + 1) ** dim
        )

        self.neighb_dice = ti.field(
            ti.i32, self.search_cell_range[None] * 2 + 1)

        for i in range(self.neighb_dice.shape[0]):
            self.neighb_dice[i] = -self.search_cell_range[None] + i

        for i in range(self.search_template.shape[0]):
            tmp = i
            for j in ti.static(range(dim)):
                digit = tmp // (self.neighb_dice.shape[0] ** (dim - j - 1))
                tmp = tmp % (self.neighb_dice.shape[0] ** (dim - j - 1))
                self.search_template[i][dim - j - 1] = self.neighb_dice[digit]

    @ti.func
    def get_neighb_cell_num(self):
        return self.search_template.shape[0]
    
    @ti.func
    def get_neighb_cell_vec(self, i):
        return self.search_template[i]
'''#################### ABOVE IS THE TEMPLATE FOR NEIGHBORHOOD SEASCHING ####################'''


'''#################### ABOVE IS THE CACHE STRUCT AND ACCOMPINED POINTER
                        STRUCT FOR LOGGING NEIGHBOUR PARTICLES ####################'''  






'''#################### BELOW IS THE CLASS FOR NEIGHBORHOOD SEASCHING ####################'''
@ti.data_oriented
class Neighb_pool(Solver):
    ''' init the neighb list'''
    def __init__(
            self,
            obj: Particle,  # Particle class
            max_neighb_part_num: ti.template() = 0,  # int32
    ):
        Solver.__init__(self, obj)
        
        if max_neighb_part_num == 0:
            self.max_neighb_part_num = val_i(obj.getObjPartNum() * self.getObj().getObjWorld().g_avg_neighb_part_num[None])
        else:
            self.max_neighb_part_num = val_i(max_neighb_part_num[None])

        self.max_neighb_obj_num = val_i(self.getObj().getObjWorld().g_obj_num[None])

        self.neighb_obj_list = []  # Particle class
        self.neighb_cell_list = []  # Neighb_cell_simple class
        self.m_neighb_search_range_list = []  # val_f() # TODO: use 'Dynamic' as search range
        self.m_neighb_search_template_list = []  # Neighb_search_template class

        self.partNeighb_begin = ti.field(ti.i32)
        self.partNeighb_current = ti.field(ti.i32)
        self.partNeighb_size = ti.field(ti.i32)
        ti.root.dense(ti.i, self.obj.getObjPartNum()).place(
            self.partNeighb_begin,
            self.partNeighb_current,
            self.partNeighb_size)

        self.partNeighbObj_begin = ti.field(ti.i32)
        self.partNeighbObj_size = ti.field(ti.i32)
        ti.root.dense(ti.ij, (self.obj.getObjPartNum(), self.max_neighb_obj_num[None])).place(
            self.partNeighbObj_begin,
            self.partNeighbObj_size)

        self.poolContainer_neighbPartId = ti.field(ti.i32)
        self.poolContainer_neighbObjId = ti.field(ti.i32)
        self.poolContainer_next = ti.field(ti.i32)
        ti.root.dense(ti.i, self.max_neighb_part_num[None]).place(
            self.poolContainer_neighbPartId,
            self.poolContainer_neighbObjId,
            self.poolContainer_next)

        self.poolCachedAttr_dist = ti.field(ti.f32)
        self.poolCachedAttr_xijNorm = ti.Vector.field(self.getObj().getObjWorld().getWorldDim(), ti.f32)
        self.poolCachedAttr_W = ti.field(ti.f32)
        self.poolCachedAttr_gradW = ti.Vector.field(self.getObj().getObjWorld().getWorldDim(), ti.f32)
        ti.root.dense(ti.i, self.max_neighb_part_num[None]).place(
            self.poolCachedAttr_dist,
            self.poolCachedAttr_xijNorm,
            self.poolCachedAttr_W,
            self.poolCachedAttr_gradW)

        self.neighb_pool_used_space = val_i(0)
        self.neighb_pool_size = ti.static(self.max_neighb_part_num)

    ''' clear the cache pool'''
    @ti.kernel
    def clear_pool(self):
        for part_id in range(self.tiGetObj().tiGetObjStackTop()):
            self.tiSet_partNeighbBeginingPointer(part_id, -1)
            self.tiSet_partNeighbCurrnetPointer(part_id, -1) 
            self.tiSet_partNeighbSize(part_id, 0)
            for obj_seq in range(self.max_neighb_obj_num[None]):
                self.tiSet_partNeighbObjBeginingPointer(part_id, obj_seq, -1)
                self.tiSet_partNeighbObjSize(part_id, obj_seq, 0)
        self.neighb_pool_used_space[None] = 0

    ''' add a $neighb obj$ to the neighb search range'''
    def add_neighb_obj(
            self,
            neighb_obj: Particle,  # Particle class
            search_range: ti.template(),  # val_f() # TODO: use 'Dynamic' as search range
    ):
        ''' check input validity'''
        if neighb_obj in self.neighb_obj_list:
            raise Exception("neighb_obj already in list")
        if neighb_obj.m_neighb_search.neighb_cell in self.neighb_cell_list:
            raise Exception("neighb_cell already in list")
        if self.getObj().getObjWorld() != neighb_obj.getObjWorld():
            raise Exception("two obj are not in the same world")

        ''' append to lists '''
        self.neighb_obj_list.append(neighb_obj)
        self.neighb_cell_list.append(neighb_obj.m_neighb_search.neighb_cell)
        self.m_neighb_search_range_list.append(search_range)

        ''' generate search template '''
        search_cell_range = int(ti.ceil(search_range[None] / neighb_obj.m_neighb_search.neighb_cell.cell_size[None]))
        neighb_search_template = Neighb_search_template(self.getObj().getObjWorld().getWorldDim(), search_cell_range)
        self.m_neighb_search_template_list.append(neighb_search_template)

    ''' get a obj, neighb_obj attributes pair  one at a time, as inputs to register_a_neighbour() '''
    def register_neighbours(
        self,
    ):
        self.clear_pool()
        for i in range(len(self.neighb_obj_list)):
            self.register_a_neighbour(self.neighb_obj_list[i], self.m_neighb_search_range_list[i][None], self.neighb_cell_list[i], self.m_neighb_search_template_list[i])
        if not self.neighb_pool_size[None] > self.neighb_pool_used_space[None]:
            raise Exception(f"neighb_pool overflow, need {self.neighb_pool_used_space[None]} but only {self.neighb_pool_size[None]}")
        # print("debug: neighb_pool_used_space_ = ", self.neighb_pool_used_space_[None], " / ", self.neighb_pool_size_[None], " = ", self.neighb_pool_used_space_[None] / self.neighb_pool_size_[None]*100, " %")
    ''' register all particles form a $neighbour obj$ to $obj particles$ as neighbours '''
    @ti.kernel
    def register_a_neighbour(
        self,
        neighbObj: ti.template(),
        search_range: ti.f32,
        neighb_cell: ti.template(),  # Neighb_cell_simple class
        neighb_search_template: ti.template(),  # Neighb_search_template class
    ):
        for part_id in range(self.tiGetObj().tiGetObjStackTop()):
            size_before = self.tiGet_partNeighbSize(part_id)

            ''' locate the cell where the $obj particle$ is located '''
            located_cell_vec = neighb_cell.compute_cell_vec(self.tiGetObj().tiGetPos(part_id))
            ''' iterate over all neighbouring cells '''
            for neighb_cell_iter in range(neighb_search_template.get_neighb_cell_num()):
                ''' get the $cell vector$ of the neighbouring cell through the template'''
                neighb_cell_vec = located_cell_vec + neighb_search_template.get_neighb_cell_vec(neighb_cell_iter)
                ''' check if the neighbouring cell is within the domain '''
                if not neighb_cell.within_cell(neighb_cell_vec):
                    continue
                ''' get the neighbouring cell id by encoding the $cell vector$ '''
                neighb_cell_id = neighb_cell.encode_cell_vec(neighb_cell_vec)
                ''' get the number of particles in the neighbouring cell '''
                part_num = neighb_cell.getObjPartNum_in_cell(neighb_cell_id)
                for part_iter in range(part_num):
                    ''' get the particle id in the neighbouring cell '''
                    neighb_part_id = neighb_cell.get_part_id_in_cell(neighb_cell_id, part_iter)
                    dist = (self.tiGetObj().tiGetPos(part_id) - neighbObj.tiGetPos(neighb_part_id)).norm()
                    ''' register the neighbouring particle '''
                    if dist < search_range:
                        pointer = self.insert_a_part(part_id, neighbObj.tiGetObjId()[None], neighb_part_id)
                        
                        ''' [DIY AREA] '''
                        ''' You can add attributes you want to be pre-computed here '''
                        self.tiSet_cachedDist(pointer, dist)
                        self.tiSet_cachedXijNorm(pointer, (self.tiGetObj().tiGetPos(part_id) - neighbObj.tiGetPos(neighb_part_id)) / dist)
                        self.tiSet_cachedW(pointer, spline_W(dist, self.tiGetObj().tiGetSphH(part_id), self.tiGetObj().tiGetSphSig(part_id)))
                        self.tiSet_cachedGradW(pointer, grad_spline_W(dist, self.tiGetObj().tiGetSphH(part_id), self.tiGetObj().tiGetSphSigInvH(part_id)) * self.tiGet_cachedXijNorm(pointer))

            self.tiSet_partNeighbObjSize(part_id, neighbObj.tiGetObjId()[None], self.tiGet_partNeighbSize(part_id) - size_before)

    ''' insert a neighbouring particle into the linked list'''
    @ti.func
    def insert_a_part(
        self,
        part_id: ti.i32,
        neighb_obj_id: ti.i32,
        neighb_part_id: ti.i32,
    ) -> ti.i32:
        pointer = ti.atomic_add(self.neighb_pool_used_space[None], 1)
        self.tiAdd_partNeighbSize(part_id, 1)
        
        if self.tiGet_partNeighbBeginingPointer(part_id) == -1:
            self.tiSet_partNeighbBeginingPointer(part_id, pointer)
        else:
            self.tiSet_nextPointer(self.tiGet_partNeighbCurrnetPointer(part_id), pointer)
        
        if self.tiGet_partNeighbObjBeginingPointer(part_id, neighb_obj_id) == -1:
            self.tiSet_partNeighbObjBeginingPointer(part_id, neighb_obj_id, pointer)

        self.tiSet_partNeighbCurrnetPointer(part_id, pointer)
        self.tiSet_neighbObjId(pointer, neighb_obj_id)
        self.tiSet_neighbPartId(pointer, neighb_part_id)

        return pointer
    
    @ti.func
    def tiGet_cachedDist(self, pointer: ti.i32):
        return self.poolCachedAttr_dist[pointer]
    @ti.func
    def tiGet_cachedGradW(self, pointer: ti.i32):
        return self.poolCachedAttr_gradW[pointer]
    @ti.func
    def tiGet_cachedW(self, pointer: ti.i32):
        return self.poolCachedAttr_W[pointer]
    @ti.func
    def tiGet_cachedXijNorm(self, pointer: ti.i32):
        return self.poolCachedAttr_xijNorm[pointer]
    @ti.func
    def tiGet_neighbPartId(self, pointer: ti.i32):
        return self.poolContainer_neighbPartId[pointer]
    @ti.func
    def tiGet_neighbObjId(self, pointer: ti.i32):
        return self.poolContainer_neighbObjId[pointer]
    @ti.func
    def tiGet_nextPointer(self, pointer: ti.i32):
        return self.poolContainer_next[pointer]
    @ti.func
    def tiGet_partNeighbObjBeginingPointer(self, partId: ti.i32, neighbObj_id: ti.i32):
        return self.partNeighbObj_begin[partId, neighbObj_id]
    @ti.func
    def tiGet_partNeighbObjSize(self, partId: ti.i32, neighbObj_id: ti.i32):
        return self.partNeighbObj_size[partId, neighbObj_id]
    @ti.func
    def tiGet_partNeighbBeginingPointer(self, partId: ti.i32):
        return self.partNeighb_begin[partId]
    @ti.func
    def tiGet_partNeighbSize(self, partId: ti.i32):
        return self.partNeighb_size[partId]
    @ti.func
    def tiGet_partNeighbCurrnetPointer(self, partId: ti.i32):
        return self.partNeighb_current[partId]

    @ti.func
    def tiSet_cachedDist(self, pointer: ti.i32, val: ti.f32):
        self.poolCachedAttr_dist[pointer] = val
    @ti.func
    def tiSet_cachedGradW(self, pointer: ti.i32, val: ti.template()):
        self.poolCachedAttr_gradW[pointer] = val
    @ti.func
    def tiSet_cachedW(self, pointer: ti.i32, val: ti.f32):
        self.poolCachedAttr_W[pointer] = val
    @ti.func
    def tiSet_cachedXijNorm(self, pointer: ti.i32, val: ti.template()):
        self.poolCachedAttr_xijNorm[pointer] = val
    @ti.func
    def tiSet_neighbPartId(self, pointer: ti.i32, val: ti.i32):
        self.poolContainer_neighbPartId[pointer] = val
    @ti.func
    def tiSet_neighbObjId(self, pointer: ti.i32, val: ti.i32):
        self.poolContainer_neighbObjId[pointer] = val
    @ti.func
    def tiSet_nextPointer(self, pointer: ti.i32, val: ti.i32):
        self.poolContainer_next[pointer] = val
    @ti.func
    def tiSet_partNeighbObjBeginingPointer(self, partId: ti.i32, neighbObj_id: ti.i32, val: ti.i32):
        self.partNeighbObj_begin[partId, neighbObj_id] = val
    @ti.func
    def tiSet_partNeighbObjSize(self, partId: ti.i32, neighbObj_id: ti.i32, val: ti.i32):
        self.partNeighbObj_size[partId, neighbObj_id] = val
    @ti.func
    def tiSet_partNeighbBeginingPointer(self, partId: ti.i32, val: ti.i32):
        self.partNeighb_begin[partId] = val
    @ti.func
    def tiSet_partNeighbSize(self, partId: ti.i32, val: ti.i32):
        self.partNeighb_size[partId] = val
    @ti.func
    def tiAdd_partNeighbSize(self, partId: ti.i32, val: ti.i32):
        ti.atomic_add(self.partNeighb_size[partId], val)    
    @ti.func
    def tiSet_partNeighbCurrnetPointer(self, partId: ti.i32, val: ti.i32):
        self.partNeighb_current[partId] = val
'''#################### ABOVE IS THE CLASS FOR NEIGHBORHOOD SEASCHING ####################'''

