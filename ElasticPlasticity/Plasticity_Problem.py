from ast import List
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

        self._init_linear_problem()
        self._init_nonlinear_problem()

        self.init_linear_solver()
        self.init_non_linear_solver()

    def _create_funcs_spaces(self):
        self.Ve = element(
            "Lagrange", self.domain.basix_cell(), self.deg_u, shape=(3,)
        )  # 2 degrees  of freedom
        self.V = functionspace(self.domain, self.Ve)

        self.Ve_scal = element("Lagrange", self.domain.basix_cell(), self.deg_u)
        self.V_scal = functionspace(self.domain, self.Ve_scal)

        self.We = quadrature_element(
            self.domain.basix_cell(),
            value_shape=(6,),
            degree=self.deg_stress,
            scheme="default",
        )
        self.W = functionspace(self.domain, self.We)

        self.W_scal_e = quadrature_element(
            self.domain.basix_cell(), degree=self.deg_stress, scheme="default"
        )
        self.W_scal = functionspace(self.domain, self.W_scal_e)

    def _create_funcs(self):
        self.E_p = Function(self.W, name="Total_Plastic_Strain")
        self.e_p = Function(self.W_scal, name="Equivalent_Plastic_Strain")
        self.u = Function(self.V, name="Total_displacement")
        self.du = Function(self.V, name="Trial_displacement")

        self.dp = Function(self.W_scal, name="Delta_plasticity")

        self.Y = Function(self.W_scal, name="Isotropic_Hardening")
        self.A = Function(self.W_scal, name="Kinematic_Hardening")

        self.Y.interpolate(lambda x: np.full_like(x[0], self.mech["sig0"]))

        self.v = TestFunction(self.V)  # Function we are testing with
        self.du_ = TrialFunction(self.V)  # Function we are solving for

        self.e_p_ = TrialFunction(self.W_scal)
        self.e_pv = TestFunction(self.W_scal)

    def _init_linear_problem(self, bc_neumann: list = None):
        E_n = eps(self.u)
        del_E_n = eps(self.du_)
        E_p_tensor = as_3D_tensor(self.E_p)
        E_e_trial = E_n - E_p_tensor + del_E_n  # Trial Elastic Strain

        T_trial = self.mat.sigma(E_e_trial)  # Trial cauchy stress
        # bilinear Part to solve for du Incrementally
        # This is the linear part (In total this will be 0)
        F_body = Constant(self.domain, np.array((0.0, 0.0, 0.0)))

        F = ufl.inner(T_trial, eps(self.v)) * self.dx - ufl.inner(F_body, self.v) * dx

        if bc_neumann is not None:
            for boundary_condition in bc_neumann:
                F += ufl.inner(boundary_condition[0], self.v) * boundary_condition[1]

        self.a_du, self.L_du = fem.form(ufl.lhs(F)), fem.form(ufl.rhs(F))

    def _init_nonlinear_problem(
        self, bc_neumann: list[tuple[ufl.Form, ufl.Measure]] = None
    ):
        E_n = eps(self.u)
        E_p_tensor = as_3D_tensor(self.E_p)
        del_E = eps(self.du)
        E_e_trial_plastic = (
            E_n - E_p_tensor + del_E
        )  # Trial Elastic Strain for plastic step

        T_trial_p = self.mat.sigma(E_e_trial_plastic)

        sigma_vm_trial = normVM(T_trial_p)  # Trial Von Mises Stress
        f_trial = sigma_vm_trial - self.Y  # Trial Yield Function

        Phi = sigma_vm_trial - 3 * self.mat.mu * self.dp - self.Y - self.Y_dot(self.dp)
        Phi_cond = conditional(gt(f_trial, 0), Phi, self.dp)  # Plastic multiplier

        self.res_p = inner(Phi_cond, self.e_pv) * self.dx

        if bc_neumann is not None:
            for boundary_condition in bc_neumann:
                F += ufl.inner(boundary_condition[0], self.v) * boundary_condition[1]
        self.N_p = dev(T_trial_p) / sigma_vm_trial

        self.Jacobian = derivative(self.res_p, self.dp, self.e_p_)

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

    def init_linear_solver(self):
        """Initializes the linear solver.

        Parameters:
        -----------
            bcs (List): List of boundary conditions.
        """
        self.A = assemble_matrix(self.a_du)
        self.A.assemble()
        self.b = create_vector(self.L_du)

        self.linear_solver = PETSc.KSP().create(self.domain.comm)
        self.linear_solver.setOperators(self.A)
        self.linear_solver.setType(PETSc.KSP.Type.PREONLY)
        self.linear_solver.getPC().setType(PETSc.PC.Type.LU)

    def solve_linear(self, bcs=None):
        """Solves the linear problem.

        Parameters:
        -----------
            bcs (List): List of boundary conditions.

        Returns:
        --------
            Function: Displacement function.
        """
        with self.b.localForm() as loc_L:
            loc_L.set(0)
        self.A.zeroEntries()
        assemble_matrix(self.A, self.a_du, bcs=bcs)
        self.A.assemble()
        assemble_vector(self.b, self.L_du)

        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        apply_lifting(self.b, [self.a_du], [bcs], [self.u.vector], scale=1)

        set_bc(self.b, bcs, self.u.vector)

        self.linear_solver.solve(self.b, self.du.vector)

    def init_non_linear_solver(self, bcs=None):
        problem = NonlinearProblem(self.res_p, self.dp, [bcs], self.Jacobian)
        self.nls_solver = NewtonSolver(MPI.COMM_WORLD, problem)
        # nls_solver.convergence_criterion = "incremental"
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
        # opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
        # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
        opts[f"{option_prefix}ksp_max_it"] = 50
        ksp.setFromOptions()

    def solve_nls(self, bcs=None):
        self.nls_solver.solve(self.dp)
        # Do updates
        e_p_exp = Expression(
            self.e_p + self.dp, self.W_scal.element.interpolation_points()
        )
        self.e_p.interpolate(e_p_exp)

        E_p_expr = Expression(
            self.E_p + tensor_to_vector(self.N_p * self.dp * sqrt(3 / 2)),
            self.W.element.interpolation_points(),
        )

        self.E_p.interpolate(E_p_expr)

        Y_exp = Expression(
            self.Y + self.Y_dot(self.dp), self.W_scal.element.interpolation_points()
        )
        self.Y.interpolate(Y_exp)
