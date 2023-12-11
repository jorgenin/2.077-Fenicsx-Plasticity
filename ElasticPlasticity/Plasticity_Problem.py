from ast import List
from typing import Dict
from matplotlib.pylab import beta
from ufl.tensors import ComponentTensor
import numpy as np

import dolfinx

from mpi4py import MPI
from petsc4py import PETSc
import basix.ufl
from dolfinx import fem, mesh, io, plot, log, default_scalar_type
from dolfinx.fem import Constant, dirichletbc, Function, functionspace, Expression
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile, VTXWriter
import ufl
from ufl import (
    Jacobian,
    TestFunctions,
    TrialFunction,
    Identity,
    eq,
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
    ge,
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

from ElasticPlasticity.Plasticity_Funcs import (
    eps,
    as_3D_tensor,
    normVM,
    tensor_to_vector,
)

from ElasticPlasticity.Plastic_material import Plastic_Material


""" ______________________________________________________________________________________________________________________"""


class Plastic_Problem_Def:
    def __init__(
        self,
        domain: mesh.Mesh,
        material: Plastic_Material,
        deg_u: int = 2,
        deg_stress: int = 2,
        facet_tags=None,
    ):
        self.domain = domain
        self.mat = material

        self.n = ufl.FacetNormal(domain)

        self.dx = ufl.Measure(
            "dx",
            domain=domain,
            metadata={"quadrature_degree": deg_u, "quadrature_scheme": "default"},
        )

        self.ds = ufl.Measure(
            "ds",
            domain=domain,
            subdomain_data=facet_tags,
            metadata={"quadrature_degree": deg_u},
        )

        self.deg_u = deg_u
        self.deg_stress = deg_stress

        self.facet_tags = facet_tags
        self._create_funcs_spaces()
        self._create_funcs()

    def _create_funcs_spaces(self):
        self.Ve = element(
            "Lagrange", self.domain.basix_cell(), self.deg_u, shape=(3,)
        )  # 2 degrees  of freedom

        self.We = quadrature_element(
            self.domain.basix_cell(),
            degree=self.deg_stress,
            scheme="default",
        )

        self.W_block_ele = basix.ufl.blocked_element(
            self.We, shape=(3, 3), symmetry=True
        )

        self.ME = mixed_element([self.Ve, self.W_block_ele, self.We, self.W_block_ele])

        self.W_space = functionspace(self.domain, self.ME)

    def _create_funcs(self):
        self.W = Function(self.W_space, name="W")
        self.W0 = Function(self.W_space, name="W0")
        self.u, self.E_p, self.Y, self.A = self.W.split()
        self.u0, self.E_p0, self.Y0, self.A0 = self.W0.split()

        # Initialize the problem

        # interpolating into Y
        self.Y.interpolate(lambda x: np.full_like(x[0], self.mat.sig0))
        self.Y0.interpolate(lambda x: np.full_like(x[0], self.mat.sig0))

    def init_nonlinear_problem(
        self, bc_neumann: list[tuple[ufl.Form, ufl.Measure]] = None
    ):
        u_test, E_p_test, Y_test, A_test = TestFunctions(
            self.W_space
        )  # Function we are testing with

        W_trial = TrialFunction(self.W_space)
        u, E_p, Y, A = ufl.split(self.W)
        u0, E_p0, Y0, A0 = ufl.split(self.W0)

        E = eps(u)
        E0 = eps(u0)

        E_e = E - E_p  # elastic stress

        Stress = self.mat.sigma(E_e)

        d_E_p = E_p - E_p0  # difference in E_p
        E_dot = E - E0  # difference in E
        Stress_eff = dev(Stress) - self.mat.C * A
        Stress_VM = normVM(Stress_eff)
        N_p = conditional(
            eq(Stress_VM, 0), 0 * Stress_eff, sqrt(3 / 2) * Stress_eff / Stress_VM
        )

        f = Stress_VM - Y  # Relation

        d_Y = Y - Y0  # change in Y

        e_p = sqrt(2 / 3) * sqrt(inner(d_E_p, d_E_p))

        H = self.Y_dot(e_p) + self.mat.C * (
            3 / 2 - sqrt(3 / 2) * self.mat.gamma * inner(A, N_p)
        )

        H = 0
        Beta = 3 * self.mat.mu / (3 * self.mat.mu + H)

        A_dot = A - A0

        chi = conditional(ge(f, 0), 1, 0)

        Res_1 = inner(Stress, eps(u_test)) * self.dx  # Stress Relation
        Res_2 = (
            inner(d_Y - self.Y_dot(e_p), Y_test) * self.dx
        )  # Isotropic Hardening Relation
        Res_3 = (
            inner(d_E_p - chi * Beta * inner(N_p, E_dot) * N_p, E_p_test) * self.dx
        )  # Change in plastic strain

        Res_4 = inner(A_dot - d_E_p + self.mat.gamma * A * e_p, A_test) * self.dx

        self.Res = (
            Res_1 + inner(Y, Y_test) * self.dx + inner(A, A_test) * self.dx
        ) + Res_3

        self.jac = derivative(self.Res, self.W, W_trial)

    def Y_dot(self, e_p: Function) -> Function:
        """Returns the derivative of the isotropic hardening function.

        Parameters:
        -----------
            e_p (Function): Equivalent plastic strain.

        Returns:
        --------
            Function: Derivative of the isotropic hardening function.
        """

        H_val = self.mat.H_0 * (1 - self.Y / self.mat.Y_s) ** self.mat.r
        return H_val * e_p

    def init_non_linear_solver(self, bcs=None):
        if bcs is None:
            bcs = []
        problem = NonlinearProblem(self.Res, self.W, bcs, self.jac)
        self.nls_solver = NewtonSolver(MPI.COMM_WORLD, problem)
        self.nls_solver.convergence_criterion = "incremental"
        self.nls_solver.rtol = 1e-8
        self.nls_solver.atol = 1e-8
        self.nls_solver.max_it = 50
        self.nls_solver.report = True
        self.nls_solver.relaxation_parameter = 1

        ksp = self.nls_solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "preonly"
        opts[f"{option_prefix}pc_type"] = "lu"
        opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
        opts[f"{option_prefix}ksp_max_it"] = 50
        ksp.setFromOptions()

    def solve_nls(self):
        self.nls_solver.solve(self.W)

        vec1 = fem.assemble_vector(fem.form(self.Res))

        print(f"Residual: {vec1.norm()}")
        # Do updates
        self.W.x.scatter_forward()
        self.W0.x.array[:] = self.W.x.array
