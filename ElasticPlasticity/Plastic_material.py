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


class Plastic_Material:
    def __init__(
        self,
        domain,
        E: float,
        nu: float,
        sig0: float,
        H: float,
        r: float,
        Y_s: float,
        C: float,
        gamma: float,
    ) -> None:
        self.E = Constant(domain, E)
        self.nu = Constant(domain, nu)
        self.lmbda = self.E * self.nu / (1 + self.nu) / (1 - 2 * self.nu)
        self.mu = self.E / 2.0 / (1 + self.nu)
        self.sig0 = Constant(domain, sig0)  # yield strength
        self.H_0 = Constant(domain, H)  # hardening modulus
        self.r = Constant(domain, r)  # hardening modulus
        self.Y_s = Constant(domain, Y_s)  # hardening modulus
        self.C = Constant(domain, C)
        self.domain = domain
        self.gamma = Constant(domain, gamma)

    def sigma(self, eps_el: Function) -> ComponentTensor:
        return self.lmbda * tr(eps_el) * Identity(3) + 2 * self.mu * eps_el
