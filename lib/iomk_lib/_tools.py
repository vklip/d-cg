from scipy.fftpack import fft, ifft, ifftshift
from scipy.integrate import cumtrapz
import scipy.optimize
from numba import njit
import h5py
import numpy as np

__all__ = [
    "_get_dim_full",
    "_get_dim_dosz",
    "_dosz_normal_to_param",
    "_mat_to_param",
    "_mat_to_param_dosz",
    "_param_to_mat",
    "_param_to_mat_dosz",
    "_wrap_func",
    "build_arff",
    "make_a_mat_function",
    "make_a_mat_function_k",
    "_calc_gt",
    "_calc_kt",
    "make_dosz_function",
    "make_dosz_function_k",
    "normalize_a_matrix",
    "normalize_g",
    "normalize_k",
    "write_a_matrix_full",
    "write_a_matrix_dosz",
]


def _get_dim_full(n):
    """get dimension of a drift matrix from number of parameters

    Args:
        n (_type_): _description_

    Returns:
        int:  Dimension of  drift matrix
    """

    dim = np.sqrt(2 * (n + 1) + 0.25) - 0.5

    # Check if n is valid
    if n < 2:
        raise ValueError("Invlid number of parameters for full matrix: choose n >= 2!")
    elif not dim / int(dim) == 1:
        raise ValueError(
            f"Invlid number of parameters for full matrix: resulting dimensionality is non-integer {dim}."
        )
    else:
        dim = int(dim)
    return dim


def _get_dim_dosz(n):
    dim = n / 2 + 1
    if n < 4:
        raise ValueError(
            "Invlid number of parameters for damped oscillator matrix: choose n >= 4!"
        )
    elif not dim / int(dim) == 1:
        raise ValueError(
            f"Invlid number of parameters for damped oscillator matrix: resulting dimensionality is non-integer {dim}."
        )
    else:
        dim = int(dim)
    return dim


def _get_n_param(dim):
    n = 0.5 * (dim) * (dim + 1) - 1
    if dim < 1:
        raise ValueError("Invlid matrix dimension: choose dim >= 2!")
    elif not n / int(n) == 1:
        raise ValueError(
            f"Invlid matrix dimension: resulting number of parameters is non-integer {n}."
        )
    n = int(n)
    return n


def _get_n_param_dosz(dim):
    n = (dim - 1) * 2
    if dim < 1:
        raise ValueError("Invlid matrix dimension: choose dim >= 2!")
    elif (dim - 1) % 2 != 0:
        raise ValueError("Invlid matrix dimension: dim has to be uneven!")
    elif not n / int(n) == 1:
        raise ValueError(
            f"Invlid matrix dimension: resulting number of parameters is non-integer {n}."
        )
    n = int(n)
    return n


"""functions from parameters"""


def _wrap_func(func, time):
    """
    Wraps a function of a memory kernel for given time array such that a call only needs the set of parameters.

    """

    def func_w(param):
        return func(time, param)

    return func_w


# Full matrix for G
def make_a_mat_function(n):
    """Generate Function to evaluate integrated memory kernel from time series and parameters.

    Args:
        n (Integer): Dimension of drift matrix (n times n)

    Returns:
        callable:
    """
    A_list = np.zeros((1, 1, 1))

    def func(time, param):
        dt = time[1] - time[0]
        nt = len(time)
        amat = _param_to_mat(param)
        assert n == len(amat)
        A_xy = amat[0, 1:]
        A_yy = amat[1:, 1:]
        A_yy_inv = np.linalg.inv(A_yy)
        A_yy_exp = scipy.linalg.expm(-A_yy * dt)
        if (
            A_list.shape[0] != nt
            or A_list.shape[1] != A_yy_exp.shape[0]
            or A_list.shape[2] != A_yy_exp.shape[1]
        ):
            print("resizing A_list")
            A_list.resize(
                (nt, A_yy_exp.shape[0], A_yy_exp.shape[1]), refcheck=False
            )  # added refcheck = False to prevent error massage on some machines.
        gt = (A_xy @ A_yy_inv @ A_xy.transpose()) * np.ones(nt)
        _calc_gt(gt, A_xy, A_yy_inv, A_yy_exp, nt, A_list)
        return gt

    return func


# free matrix for K
def make_a_mat_function_k(n):
    """Generate Function to evaluate memory kernel (not integrated) from time series and parameters.

    Args:
        n (Integer): Dimension of drift matrix (n times n)

    Returns:
        callable:
    """
    A_list = np.zeros((1, 1, 1))

    def func(time, param):
        dt = time[1] - time[0]
        nt = len(time)
        amat = _param_to_mat(param)
        assert n == len(amat)
        A_xy = amat[0, 1:]
        A_yy = amat[1:, 1:]
        A_yy_exp = scipy.linalg.expm(-A_yy * dt)
        if (
            A_list.shape[0] != nt
            or A_list.shape[1] != A_yy_exp.shape[0]
            or A_list.shape[2] != A_yy_exp.shape[1]
        ):
            print("resizing A_list")
            A_list.resize(
                (nt, A_yy_exp.shape[0], A_yy_exp.shape[1]), refcheck=False
            )  # added refcheck = False to prevent error massage on some machines.
        kt = np.zeros(nt)
        _calc_kt(kt, A_xy, A_yy_exp, nt, A_list)
        return kt

    return func


@njit
def _calc_gt(gt, A_xy, A_yy_inv, A_yy_exp, nt, A_list):
    """Evaluates integrated memory kernel from drift matrix

    Args:
        gt numpy.array: the results (G(t)) are stored here.
        A_xy numpy.array:
        A_yy_inv numpy.array:
        A_yy_exp numpy.array:
        nt int: _description_
        A_list list:
    """
    A_list[0] = np.identity(len(A_yy_exp))
    A_list[1] = A_yy_exp
    A_xy_Ayy_inv = A_xy @ A_yy_inv
    A_xy_t = A_xy.transpose()
    gt[0] -= A_xy_Ayy_inv @ A_list[0] @ A_xy_t
    gt[1] -= A_xy_Ayy_inv @ A_list[1] @ A_xy_t

    for i in range(2, nt):
        A_list[i] = A_list[i // 2] @ A_list[(i + 1) // 2]
        # gt[i] -= A_xy@A_yy_inv@A_list[i]@A_xy.transpose()
        gt[i] -= A_xy_Ayy_inv @ A_list[i] @ A_xy_t


@njit
def _calc_kt(kt, A_xy, A_yy_exp, nt, A_list):
    """Evaluates memory kernel from drift matrix"""
    A_list[0] = np.identity(len(A_yy_exp))
    A_list[1] = A_yy_exp
    A_xy_t = A_xy.transpose()
    for i in range(nt):
        A_list[i] = A_list[i // 2] @ A_list[(i + 1) // 2]
        kt[i] = A_xy @ A_list[i] @ A_xy_t


def make_dosz_function(dim):  # TODO Not in use, Check implementation before usage
    """Generate callable function for dampened oscillator"""
    n_osc = (dim - 1) // 2

    def func(x, param):
        out = np.zeros(len(x))
        for i in range(n_osc):
            a = param[i * 4]
            e = param[i * 4 + 1] ** 2
            f = param[i * 4 + 2] ** 2
            d = param[i * 4 + 3]
            c = a * f / 2.0 / d - a * e / d / 2.0
            b = 2.0 * e + 2 * c * d / a

            out += -2.0 * np.exp(-0.5 * a * x) * (
                np.sin(d * x) * (a * c - 2.0 * b * d)
                + np.cos(d * x) * (a * b + 2.0 * c * d)
            ) / (a**2.0 + 4.0 * d**2.0) + 2.0 * ((a * b + 2.0 * c * d)) / (
                a**2.0 + 4.0 * d**2.0
            )

        return out

    return func


def make_dosz_function_k(dim):
    n_osc = (dim - 1) // 2

    def func(x, param):
        # _apply_dosz_constraints(param, dim)
        out = np.zeros(len(x))
        for i in range(n_osc):
            a = param[i * 4]
            e = param[i * 4 + 1] ** 2
            f = param[i * 4 + 2] ** 2
            d = param[i * 4 + 3]
            c = a * f / 2.0 / d - a * e / d / 2.0
            b = 2.0 * e + 2 * c * d / a

            out += np.exp(-0.5 * a * x) * (c * np.sin(d * x) + b * np.cos(d * x))
        return out

    return func


"""param <-----> mat"""


# @njit
def _param_to_mat(param):
    """Translates a list of parameters to full drift matrix

    Args:
        param (_type_): _description_
        n (_type_): _description_

    Returns:
        _type_: _description_
    """
    n = _get_dim_full(len(param))
    mat = np.zeros((n, n))
    p_index = 0
    for i in range(1, n):
        mat[0][i] = param[p_index]
        mat[i][0] = -param[p_index]
        p_index += 1
        for j in range(i, n):
            mat[i][j] = param[p_index]
            if i != j:
                mat[j][i] = -param[p_index]
            p_index += 1
    # param = _apply_mat_constraints(param, n)
    return mat


def _mat_to_param(mat):
    """Translates an unrestrained drift matrix to list of parameters"""
    dim = len(mat)
    param = []
    for i in range(1, dim):
        param.append(mat[0][i])
        for j in range(i, dim):
            param.append(mat[i][j])
    return np.array(param)


# Kind of dubious, if matrix was not originally built as dosz. One should be carful using this.
def _mat_to_param_dosz(A):
    dim = len(A)
    n_osc = (dim - 1) // 2
    params = []
    for i in range(n_osc):
        params.append(A[1 + i * 2, 1 + i * 2])
        params.append(A[0, 1 + i * 2])
        params.append(A[0, 2 + i * 2])
        params.append(
            np.sqrt(
                0.25
                * ((2 * A[1 + i * 2, 2 + i * 2]) ** 2 - A[1 + i * 2, 1 + i * 2] ** 2)
            )
        )
    return np.array(params)


def write_a_matrix_full(param, amat_file, renorm=[1, 1]):
    """Write unrestrained drift matrix to file"""
    n = _get_dim_full(len(param))
    mat = np.zeros((n, n))
    p_index = 0
    for i in range(1, n):
        mat[0][i] = param[p_index] * np.sqrt(renorm[1] / renorm[0])
        mat[i][0] = -param[p_index] * np.sqrt(renorm[1] / renorm[0])
        p_index += 1
        for j in range(i, n):
            mat[i][j] = param[p_index] / renorm[0]
            if i != j:
                mat[j][i] = -param[p_index] / renorm[0]
            p_index += 1

    np.savetxt(amat_file, mat)
    return mat


def _param_to_mat_dosz(params):
    def gle_param(a, e, f, d):
        ao = e
        bo = f
        co = a
        do = 0.5 * np.sqrt(4 * d**2.0 + a**2.0)

        return (ao, bo, co, do)

    n_osc = _get_dim_dosz(len(params))
    fullA = np.zeros((n_osc * 2 + 1, n_osc * 2 + 1))
    fsum = np.sum(params[2::4])
    for i in range(n_osc):
        a, b, c, d = gle_param(
            params[i * 4], params[i * 4 + 1], params[i * 4 + 2], params[i * 4 + 3]
        )
        fullA[0, 1 + i * 2] = a
        fullA[0, 2 + i * 2] = b
        fullA[1 + i * 2, 0] = -a
        fullA[2 + i * 2, 0] = -b
        fullA[1 + i * 2, 1 + i * 2] = c
        fullA[1 + i * 2, 2 + i * 2] = d
        fullA[2 + i * 2, 1 + i * 2] = -d
    return fullA
    # print(fullA)
    # np.savetxt(folder+"/Amatrix",fullA)


def write_a_matrix_dosz(params, amat_file, renorm=[1, 1]):
    def gle_param(a, e, f, d):
        ao = e  # np.sqrt(0.5*b-c*d/a)
        bo = f  # np.sqrt(0.5*b+c*d/a)
        co = a
        do = 0.5 * np.sqrt(4 * d**2.0 + a**2.0)

        # A = np.array([ [0.0, ao, bo],
        #           [-ao, co, do ],
        #           [-bo, -do, 0.0]])
        # print(A)
        return (ao, bo, co, do)

    dim = _get_dim_dosz(len(params))
    print(dim)
    n_osc = (dim - 1) // 2
    assert len(params) == n_osc * 4
    n = int(len(params) / 4)
    fullA = np.zeros((n_osc * 2 + 1, n_osc * 2 + 1))
    fsum = np.sum(params[2::4])
    for i in range(n_osc):
        a, b, c, d = gle_param(
            params[i * 4], params[i * 4 + 1], params[i * 4 + 2], params[i * 4 + 3]
        )
        fullA[0, 1 + i * 2] = a * np.sqrt(renorm[1] / renorm[0])
        fullA[0, 2 + i * 2] = b * np.sqrt(renorm[1] / renorm[0])
        fullA[1 + i * 2, 0] = -a * np.sqrt(renorm[1] / renorm[0])
        fullA[2 + i * 2, 0] = -b * np.sqrt(renorm[1] / renorm[0])
        fullA[1 + i * 2, 1 + i * 2] = c / renorm[0]
        fullA[1 + i * 2, 2 + i * 2] = d / renorm[0]
        fullA[2 + i * 2, 1 + i * 2] = -d / renorm[0]
    np.savetxt(amat_file, fullA)
    return fullA


# For convenience, work directly with analytical dosz parameters
def _dosz_normal_to_param(p):
    return np.array(
        [
            p[0],
            np.sqrt(p[1] / 2 - p[2] * p[3] / p[0]),
            np.sqrt(p[1] / 2 + p[2] * p[3] / p[0]),
            p[3],
        ]
    )


#### Auxiliary tools #####
def build_arff(int_mem_k, timestep):
    arff = np.zeros((2, len(int_mem_k)))
    for i in range(len(int_mem_k)):
        arff[0][i] = timestep * i
        arff[1][i] = int_mem_k[i]
    return arff


def normalize_a_matrix(a_mat, renorm):
    mat = a_mat.copy()
    for i in range(1, len(a_mat)):
        mat[i, 0] *= np.sqrt(renorm[0] / renorm[1])
        mat[0, i] *= np.sqrt(renorm[0] / renorm[1])

        for j in range(1, len(a_mat)):
            mat[i][j] *= renorm[0]

    return mat[:]


def normalize_g(arff):
    """Normalize integrated memory kernel G(t)

    Args:
        arff (NDarray[float]):

    Returns:
        (list,list): ([normalized t, normalized G], normalization factors)
    """
    tmax = max(arff[0])
    g_max = max(arff[1])
    return [arff[0] / tmax, arff[1] / g_max], [tmax, g_max]


def normalize_k(arff):
    """Normalize  (non-integrated) memory kernel K(t)

    Args:
        arff (NDarray[float]):

    Returns:
        (list,list): ([normalized t, normalized K], normalization factors)
    """
    g = cumtrapz(arff[1], arff[0], initial=0)
    g_norm, renorm = normalize_g(np.array([arff[0], g]))
    target_norm = [g_norm[0], arff[1] / renorm[1] * renorm[0]]
    return target_norm, renorm
