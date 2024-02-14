import numpy as np

from iomk_lib._tools import (
    _get_dim_dosz,
    _get_dim_full,
    make_a_mat_function,
    make_a_mat_function_k,
    make_dosz_function,
    make_dosz_function_k,
    normalize_g,
    normalize_k,
    _mat_to_param,
    _mat_to_param_dosz,
    _param_to_mat,
    _param_to_mat_dosz,
    write_a_matrix_full,
    write_a_matrix_dosz,
    _dosz_normal_to_param,
)


__all__ = [
    "_apply_mat_constraints",
    "_apply_dosz_constraints",
    "_guess_mat",
    "_guess_mat_dosz",
    "reg_adaptive",
    "reg_tikhonov",
    "_reg_methods",
    "_target_type_methods_full",
    "_target_type_methods_dosz",
    "_wrapped_methods",
]

# Constraints for fitting, assume normalized target


def _apply_dosz_constraints(param, n, step=[], off_diag=1e-5, diag=2):
    if len(step) == 0:
        temp_param = np.array(param)
        for i in range(len(temp_param) // 4):
            if temp_param[i * 4] < diag:
                temp_param[i * 4] = diag
            if temp_param[i * 4 + 1] < off_diag:
                temp_param[i * 4 + 1] = off_diag
            if temp_param[i * 4 + 2] < off_diag:
                temp_param[i * 4 + 2] = off_diag
            if temp_param[i * 4 + 3] < off_diag:
                temp_param[i * 4 + 3] = off_diag
        return temp_param
    else:
        temp_param = param[:] - step
        for i in range(len(temp_param) // 4):
            if temp_param[i * 4] < diag:
                temp_param[i * 4] = diag
            if temp_param[i * 4 + 1] < off_diag:
                temp_param[i * 4 + 1] = off_diag
            if temp_param[i * 4 + 2] < off_diag:
                temp_param[i * 4 + 2] = off_diag
            if temp_param[i * 4 + 3] < off_diag:
                temp_param[i * 4 + 3] = off_diag
        return temp_param


def _apply_mat_constraints(param, n, step=[], off_diag=1e-5, diag=2):
    """_summary_

    Args:
        param (iterable): _description_
        n (integer): _description_
        step (list, optional): _description_. Defaults to [].
        off_diag (float, optional): _description_. Defaults to 1e-5.
        diag (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    if len(step) == 0:
        temp_param = param[:]
        p_index = 0
        for i in range(1, n):
            if temp_param[p_index] < off_diag:
                temp_param[p_index] = off_diag

            p_index += 1
            for j in range(i, n):
                if i != j:
                    if temp_param[p_index] < off_diag:
                        temp_param[p_index] = off_diag
                else:
                    if temp_param[p_index] < diag:
                        temp_param[p_index] = diag
                p_index += 1

        return temp_param
    else:
        temp_param = param[:] - step
        p_index = 0
        for i in range(1, n):
            if temp_param[p_index] < off_diag:
                temp_param[p_index] = off_diag

            p_index += 1
            for j in range(i, n):
                if i != j:
                    if temp_param[p_index] < off_diag:
                        temp_param[p_index] = off_diag
                else:
                    if temp_param[p_index] < diag:
                        temp_param[p_index] = diag
                p_index += 1
        return temp_param


# from _tools import (_dosz_normal_to_param,)
""" Different functions to determine regularization """

# Tikhonov regularization, just choose constant regularization parameter


def reg_tikhonov(lamb):
    def reg(*args):  # allow dummy arguments to allow other regularization schemes
        return lamb

    return reg


# Solving (A+B)*delta = b, where B is a diagonal matrix with a unique entry per parameter
# This form makes sure that delta_i <= p0_i*max_rel_step for the ith parameter


def reg_adaptive(lamb):
    def reg(b, p0):
        return np.diag(np.abs(b * lamb / (p0)))

    return reg


""" Optimized initial guess for normalized fitting"""


def _guess_mat(dim, nframes, min_diag=2, max_diag_scale=0.1):
    """Generates initial guess of a drift matrix of given dimensionality dim for
    the optimization of a normalized integrated memory kernel.
    The the A_ps and the diagonal elements of A_ss are chosen such that with zeros in the off diagonal elements
    the total integral of the memory kernel would be one. The off diagonal entries are then set to be non-zero.
    The diagonal elements are logarithmically equidistantly spaced within the bounds.


    Args:
        dim (_type_): dimension of drift matrix (dim x dim)
        nframes (_type_): Number of (equidistant) frames in memory kernel.
        min_diag (int, optional): Lower bounds of diagonal elements. Defaults to 2.
        max_diag_scale (float, optional): Determines the upper bounds of the diagonal elements as nframes*max_diag_scale. Defaults to 0.1.

    Returns:
        _type_: _description_
    """
    p_guess = np.zeros(int((dim) * (dim + 1) / 2 - 1))
    p_index = 0
    part_g = 1.0 / (dim - 1)
    max_diag = max_diag_scale * nframes
    diag = np.logspace(np.log10(min_diag), np.log10(max_diag), num=dim - 1)
    for i in range(1, dim):
        p_guess[p_index] = np.sqrt(part_g * diag[i - 1])
        p_index += 1
        for j in range(i, dim):
            p_guess[p_index] = diag[i - 1]
            if i != j:
                p_guess[p_index] = diag[i - 1] / ((j - i))
            p_index += 1
    return _apply_mat_constraints(p_guess, dim)


def _guess_mat_dosz(dim, nframes, min_diag=2, max_diag_scale=0.1):
    """ """
    p_guess = np.zeros((dim - 1) // 2 * 4)
    p_index = 0
    part_g = 1.0 / (dim - 1) * 2
    max_diag = max_diag_scale * nframes
    min_diag = 2 * min_diag  # Keep definition with full matrix somewhat consistent
    diag = np.logspace(np.log10(min_diag), np.log10(max_diag), num=(dim - 1) // 2)
    print(diag)
    for i in range(len(p_guess) // 4):
        print(
            _dosz_normal_to_param(
                [diag[i], np.sqrt(part_g * diag[i]), 0, diag[i] / 100]
            )
        )
        p_guess[i * 4 : i * 4 + 4] = _dosz_normal_to_param(
            [diag[i], np.sqrt(part_g * diag[i]), 0.1 * i, diag[i] / 10]
        )[:]
    print(p_guess)
    print("HEEERE")
    return _apply_dosz_constraints(p_guess, dim)


""" Here we define a dictionary to map configuration key words to the corresponding regularization method."""
_reg_methods = {"adaptive": reg_adaptive, "tikhonov": reg_tikhonov}


##############################################


_target_type_methods_full = {
    "G": [make_a_mat_function, normalize_g],
    "K": [make_a_mat_function_k, normalize_k],
}
_target_type_methods_dosz = {
    "G": [make_dosz_function, normalize_g],
    "K": [make_dosz_function_k, normalize_k],
}

_wrapped_methods = {
    "full_matrix": {
        "mat_to_param": _mat_to_param,
        "param_to_mat": _param_to_mat,
        "guess_mat": _guess_mat,
        "target_type_methods": _target_type_methods_full,
        "get_dim": _get_dim_full,
        "constraints": _apply_mat_constraints,
        "write_a_matrix": write_a_matrix_full,
    },
    "dosz": {
        "mat_to_param": _mat_to_param_dosz,
        "param_to_mat": _param_to_mat_dosz,
        "guess_mat": _guess_mat_dosz,
        "target_type_methods": _target_type_methods_dosz,
        "get_dim": _get_dim_dosz,
        "constraints": _apply_dosz_constraints,
        "write_a_matrix": write_a_matrix_dosz,
    },
}
