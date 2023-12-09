from ufl.tensors import ComponentTensor


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


def sigma_tr(eps_el):
    """
    Generates the stress tensor from the input of a plain strain tensor.

    Parameters:
    ----------
        eps_el (numpy.ndarray): Plain strain tensor.

    Returns:
    --------
        3x3 Tensor: the trace of the plain stress tensor
    """
    return 1.0 / 3.0 * (3.0 * lmbda + 2.0 * mu) * ufl.tr(eps_el) * ufl.Identity(3)


def sigma_dev(eps_el):
    """
    Generates the stress tensor from the input of a plain strain tensor.

    Parameters:
    ----------
        eps_el (numpy.ndarray): Plain strain tensor.

    Returns:
    --------
        3x3 Tensor: The deviatoric part of the stress tensor.
    """
    return 2.0 * mu * ufl.dev(eps_el)


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


def sigma(eps_el: Function) -> ComponentTensor:
    return lmbda * tr(eps_el) * Identity(3) + 2 * mu * eps_el
