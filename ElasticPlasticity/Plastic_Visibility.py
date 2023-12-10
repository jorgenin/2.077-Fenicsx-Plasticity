from typing import Dict
from ufl.tensors import ComponentTensor
import numpy as np

import dolfinx

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh, io, plot, log, default_scalar_type
from dolfinx.fem import Constant, dirichletbc, Function, functionspace, Expression
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile, VTXWriter
import ufl
from ufl import (
    TestFunctions,
    TrialFunction,
    Identity,
    grad,
    det,
    div,
    dev,
    inv,
    tr,
    sqrt,
    conditional,
    gt,
    dx,
    inner,
    derivative,
    dot,
    ln,
    split,
    TestFunction,
    indices,
    as_tensor,
)
from basix.ufl import element, mixed_element, quadrature_element
from datetime import datetime
from dolfinx.plot import vtk_mesh
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    create_vector,
    set_bc,
    create_matrix,
    create_vector,
)
import basix

from ElasticPlasticity.Plasticity_Problem import Plastic_Problem_Def
from ElasticPlasticity.Plasticity_Funcs import (
    eps,
    as_3D_tensor,
    normVM,
    tensor_to_vector,
)


class Problem_Saver:
    def __init__(self, problem: Plastic_Problem_Def, name: str):
        UV = element("Lagrange", problem.domain.basix_cell(), 1, shape=(3,))
        UT = element(
            "Lagrange", problem.domain.basix_cell(), 1, shape=(3, 3), symmetry=True
        )
        U1 = element("Lagrange", problem.domain.basix_cell(), 1)

        VV = fem.functionspace(problem.domain, UV)  # Vector function space
        VT = fem.functionspace(problem.domain, UT)  # Vector function space
        V1 = fem.functionspace(problem.domain, U1)  # Scalar function space

        E_n = eps(problem.u)
        E_p_tensor = as_3D_tensor(problem.E_p)
        E_e_trial_plastic = E_n - E_p_tensor
        T = problem.mat.sigma(E_e_trial_plastic)

        self.u_vis = Function(VV, name="Displacement")
        self.u_expr = Expression(problem.u, VV.element.interpolation_points())

        self.T_vis = Function(VT, name="Stress")
        self.T_expr = Expression(T, VT.element.interpolation_points())

        self.E_vis = Function(VT, name="Strain")
        self.E_expr = Expression(E_n, VT.element.interpolation_points())

        self.Y_vis = Function(V1, name="Yield")
        self.Y_expr = Expression(problem.Y, V1.element.interpolation_points())

        self.E_p_vis = Function(VT, name="Plastic_Strain")
        self.E_p_expr = Expression(E_p_tensor, VT.element.interpolation_points())

        self.e_p_vis = Function(V1, name="Equivalent_Plastic_Strain")
        self.e_p_expr = Expression(problem.e_p, V1.element.interpolation_points())

        self.Mises_vis = Function(V1, name="Mises")
        self.Mises_expr = Expression(normVM(T), V1.element.interpolation_points())

        self.vtk = VTXWriter(
            problem.domain.comm,
            "results/" + name + ".bp",
            [
                self.u_vis,
                self.T_vis,
                self.E_vis,
                self.Y_vis,
                self.E_p_vis,
                self.e_p_vis,
                self.Mises_vis,
            ],
            engine="BP4",
        )

    def update_and_save(self, t):
        self.u_vis.interpolate(self.u_expr)
        self.T_vis.interpolate(self.T_expr)
        self.E_vis.interpolate(self.E_expr)
        self.Y_vis.interpolate(self.Y_expr)
        self.E_p_vis.interpolate(self.E_p_expr)
        self.e_p_vis.interpolate(self.e_p_expr)
        self.Mises_vis.interpolate(self.Mises_expr)
        self.vtk.write(t)
