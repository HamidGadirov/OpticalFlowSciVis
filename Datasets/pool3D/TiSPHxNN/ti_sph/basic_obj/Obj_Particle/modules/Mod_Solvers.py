import taichi as ti
from ....basic_solvers.Solver_adv import Adv_slover
from ....basic_solvers.Solver_df import DF_solver
from ....basic_solvers.Solver_sph import SPH_solver
from ....basic_solvers.Solver_wcsph import WCSPH_solver
from ....basic_solvers.Solver_ism import Implicit_mixture_solver
from ....basic_solvers.Solver_JL21 import JL21_mixture_solver

@ti.data_oriented
class Mod_Solvers:
    def __init__(self) -> None:
        self.m_solver_adv = None
        self.m_solver_df = None
        self.m_solver_sph = None
        self.m_solver_wcsph = None
        self.m_solver_ism = None
        self.m_solver_JL21 = None


    def add_solver_adv(self):
        self.m_solver_adv = Adv_slover(self)

    def add_solver_sph(self):
        self.m_solver_sph = SPH_solver(self)

    def add_solver_df(self, incompressible_threshold: ti.f32 = 1e-4, div_free_threshold: ti.f32 = 1e-3, incompressible_iter_max: ti.i32 = 100, div_free_iter_max: ti.i32 = 50, incomp_warm_start: bool = False, div_warm_start: bool = False):
        self.m_solver_df = DF_solver(self, incompressible_threshold, div_free_threshold, incompressible_iter_max, div_free_iter_max, incomp_warm_start, div_warm_start)

    def add_solver_wcsph(self, gamma: ti.f32 = 7, stiffness: ti.f32 = 1000):
        self.m_solver_wcsph = WCSPH_solver(self, gamma, stiffness)

    def add_solver_ism(self, Cd, Cf, k_vis_inter, k_vis_inner):
        self.m_solver_ism = Implicit_mixture_solver(self, Cd, Cf, k_vis_inter, k_vis_inner, self.m_world)

    def add_solver_JL21(self, kd, Cf, k_vis):
        self.m_solver_JL21 = JL21_mixture_solver(self, kd, Cf, k_vis, self.m_world)

    def getSolverSPH(self)->SPH_solver:
        return self.m_solver_sph
    @ti.func
    def tiGetSolverSPH(self):
        return self.m_solver_sph
    @ti.func
    def tiGetSolverAdv(self):
        return self.m_solver_adv
    def getSolverAdv(self):
        return self.m_solver_adv
    @ti.func
    def tiGetSolverDF(self):
        return self.m_solver_df
    @ti.func
    def tiGetSolverWCSPH(self):
        return self.m_solver_wcsph
    @ti.func
    def tiGetSolverISM(self):
        return self.m_solver_ism
    @ti.func
    def tiGetSolverJL21(self):
        return self.m_solver_JL21