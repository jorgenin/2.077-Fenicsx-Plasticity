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


""" 
______________________________________________________________________________________________________________________
3D Plasticity Helper Functions
----------------------------------------------------------------------------------------------------------------------
"""


def eps(v: Function) -> ComponentTensor:
    """
    Generates a  strain tensor from input of a 3D displacement vector.


    Parameters:
    -----------
        u (2d Vector Funciton): 3D displacement vector.

    Returns:
    --------
        3x3 Tensor: Plain strain tensor generated from the displacement vector.
    """
    e = ufl.sym(grad(v))
    return e  # Plain strain tensor


def as_3D_tensor(X):
    """Converts an array to a 3D tensor.

    Parameters:
    ----------
       X (Function): Array to be converted to 3D tensor.

    Returns:
    --------
        3x3 Tensor: 3D tensor generated from the input array.
    """
    return ufl.as_tensor([[X[0], X[3], X[4]], [X[3], X[1], X[5]], [X[4], X[5], X[2]]])


def tensor_to_vector(X):
    """
    Take a 3x3 tensor and return a vector of size 4 in 2D
    """
    return ufl.as_vector([X[0, 0], X[1, 1], X[2, 2], X[0, 1], X[0, 2], X[1, 2]])


def get_point(func: Function, x: float, y: float, zs: float = 0):
    """
    Returns the value of a function at a given point.

    Parameters:
    -----------
        func (Function): Function to be evaluated.
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.
        z (float): z-coordinate of the point.

    Returns:
    --------
        float: Value of the function at the given point.
    """

    return func.eval([x, y, z])


def project(v, target_func, bcs=[],dx_deg = 2 ):
    """Project UFL expression.

    Note
    ----
    This method solves a linear system (using KSP defaults).

    """

    # Ensure we have a mesh and attach to measure
    V = target_func.function_space
    dx = ufl.dx(V.mesh, metadata={"quadrature_degree": dx_deg})

    # Define variational problem for projection
    w = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)
    a = dolfinx.fem.form(ufl.inner(Pv, w) * dx)
    L = dolfinx.fem.form(ufl.inner(v, w) * dx)

    # Assemble linear system
    A = assemble_matrix(a, bcs)
    A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_func.vector)

    # Destroy PETSc linear algebra objects and solver
    solver.destroy()
    A.destroy()
    b.destroy()


def normVM(sig):  # Von Mises equivalent stress
    s_ = ufl.dev(sig)
    return ufl.sqrt(3 / 2.0 * ufl.inner(s_, s_))
