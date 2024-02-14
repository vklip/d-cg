###################################### IOMK-GN ##############################
import os
import shutil
import subprocess
import glob
from scipy.optimize._numdiff import approx_derivative
import numpy as np
from numba import njit

import iomk_lib
from iomk_lib._analyze_trj import mem_from_vacf, get_vacf
from iomk_lib._tools import normalize_a_matrix, _wrap_func, write_a_matrix_full
from iomk_lib._fit_tools import _wrapped_methods, _reg_methods, _apply_mat_constraints
from iomk_lib.fit import fit


def _check_config(config):
    difference = list(set(config) - set(iomk_lib.defaults))
    if len(difference) > 0:
        raise NotImplementedError(
            f"Configuration contains unkown top-level parameter(s): {difference}"
        )
    if not config["general"]["target_file"]:
        raise ValueError(f"Please provide a target file.")

    for k in config.keys():
        difference = list(set(config[k]) - set(iomk_lib.defaults[k]))
        if len(difference) > 0:
            raise NotImplementedError(
                f"Configuration contains unkown parameter(s): {difference} within '{k}'."
            )
        for kk in config[k].keys():
            if isinstance(config[k][kk], dict):
                _difference = list(set(config[k][kk]) - set(iomk_lib.defaults[k][kk]))
                if len(_difference) > 0:
                    raise NotImplementedError(
                        f"Configuration contains unkown parameter(s): {_difference} within '{k}.{kk}'."
                    )

                for kkk in config[k][kk].keys():
                    if not isinstance(
                        config[k][kk][kkk], type(iomk_lib.defaults[k][kk][kkk])
                    ):
                        raise ValueError(
                            f"'{k}.{kk}.{kkk}' should be of type {type(iomk_lib.defaults[k][kk][kkk])}, but an instance of {type(config[k][kk][kkk])} has been found."
                        )

            elif not isinstance(config[k][kk], type(iomk_lib.defaults[k][kk])):
                raise ValueError(
                    f"'{k}.{kk}' should be of type {type(iomk_lib.defaults[k][kk])}."
                )


def main_iomk_gn(config):
    """Runs IOMK(-GN) algorithm, using a configuration

    Args:
        config (dict): Contains all keyword arguments, forwarded to run_iomk_gn (and methods therein). See Command-line tool "iomk_gn.py" for an exemplary use case.
    """
    _check_config(config)
    general_kwargs = config["general"]
    gn_kwargs = config["iomk"]["GN"]
    iomk_kwargs = dict(config["iomk"])
    del iomk_kwargs["GN"]
    fit_config = config["fit"]
    jac_config = config["jac"]
    return run_iomk_gn(
        **general_kwargs,
        **iomk_kwargs,
        gn_kwargs=gn_kwargs,
        fit_kwargs=fit_config,
        jac_kwargs=jac_config,
    )


def run_iomk_gn(
    target_file,
    a_file,
    sim_files,
    target_type="G",
    function_type="full_matrix",
    dim=7,
    Amat_file="",
    p_guess=[],
    t_spacing=1,  #
    last_t=0,
    method="GN",
    max_it=7,
    lmp="lmp -i lmp.in",
    n_traj=1,
    traj_name="dump",
    start_it=0,
    remove_traj=False,
    tol_r=1e-5,
    targets=[],
    index_files=[],
    gn_kwargs=iomk_lib.defaults["iomk"]["GN"],
    fit_kwargs=iomk_lib.defaults["fit"],
    jac_kwargs=iomk_lib.defaults["jac"],
):
    """_summary_

    Args:
        target_file (str): path/to/file of target memory function.
        a_file (str): path/to/file of memory kernel in CG-MD simulation without dissipative thermostat. If this is not provided for IOMK(-GN), effects of conservative forces are ignored.
        sim_files (list): # List of filenames to be copied in every iteration. Typically LAMMPS scripts and data files.
        target_type (str, optional): How to interpret the target array. Options: {"G": integrated memory kernel (in frequency units), "K":memory kernel (in frequency^2 units) }. Defaults to "G".
        function_type (str, optional): # Defines the functional form of the memory kernel. Options: {"full_matrix" use a full matrix with screw-symmetric off diagonal elements, "dosz" use damped oscillators}. Defaults to "full_matrix".
        dim (int, optional): Fixes Amatrix dimension. Is overwritten if "Amat_file" or "p_guess" is provided. When neither "Amat_file" nor "p_guess" is provided, and initial guess is generically generated with dim dimensionality. Defaults to 7.
        Amat_file (str, optional): path/to/Amatrix. Used as initial guess. Gets overwritten, when p_guess is provided. CAUTION: Only provide input consistent with the chosen "function_type". Defaults to "".
        p_guess (list, optional): List of parameters. Overwrites "dim" AND "Amat_file". CAUTION: Currently assumes that parameters are already normalized. In most cases, "dim" or "Amat_file" should be used to generate initial guess.  Defaults to [].
        t_spacing (int, optional): _description_. Defaults to 1.
        last_t (int, optional): Only fit up to this entry. If 0, considers full input. Defaults to 0.
        method (str, optional):Options: {"GN": run IOMK-GN, "N": run ('normal') IOMK. Uses fit function, controlled via fit_kwargs and jac_kwargs} Defaults to "GN".
        max_it (int, optional): Maximum number of iterations steps. Defaults to 7.
        lmp (str, optional): Command to use to run simulation. When using e.g. mpirun for parallel simulations, change accordingly. Defaults to "lmp -i lmp.in".
        n_traj (int, optional): When several (independent but otherwise equivalent) trajectories are produced in one iteration, n_traj has to be set to average over all trajectories. Defaults to 1.
        traj_name (str, optional): Path/to/traj.h5 Lammps produces. When n_traj > 1, trajectories should be named i_${traj_name}. Defaults to "dump".
        start_it (int, optional): 0 means, start from scratch. If > 0, expects Amatrix in it{start_it}_mem_fit to exist  and will take matrix from there as initial guess. Defaults to 0.
        remove_traj (bool, optional): Set to True to delete trajectories between iterations.  . Defaults to False.
        tol_r (_type_, optional):Residue tolerance. Optimization stops when mean squared error is below tol_r. Defaults to 1e-5.
        targets (list, optional): TODO, placeholder not yet implemented. Defaults to [].
        index_files (list, optional): TODO, placeholder not yet implemented. Defaults to [].
        gn_kwargs (_type_, optional): If method_type="GN", these kwargs are forwarded to the IOMK-Gauss-Newton update step. Defaults to iomk_lib.defaults["iomk"]["GN"].
        fit_kwargs (_type_, optional):  If method_type="N", these kwargs are forwarded to the fit function in the IOMK update step.. Defaults to iomk_lib.defaults["fit"].
        jac_kwargs (_type_, optional): Forwarded to scipy.optimize._numdiff.approx_derivative. Defaults to iomk_lib.defaults["jac"].

    """
    dir_path = str(subprocess.check_output("pwd", shell=True)[:-1])[2:-1] + "/"
    os.chdir(dir_path)

    if start_it > 0:
        if os.path.isfile(dir_path + f"it{start_it}_mem_fit/Amatrix"):
            Amat_file = dir_path + f"it{start_it}_mem_fit/Amatrix"
            p_guess = []
        else:
            raise FileNotFoundError(
                f"File 'it{start_it}_mem_fit/Amatrix' does not exist. Optimization cannot be extended."
            )

    target = np.loadtxt(target_file)
    a_mem = np.loadtxt(a_file)
    wrapped_methods = _wrapped_methods[function_type]

    if function_type not in _wrapped_methods.keys():
        raise ValueError(f'Unkown function_type "{function_type}".')

    else:
        mat_to_param = wrapped_methods["mat_to_param"]
        param_to_mat = wrapped_methods["param_to_mat"]
        guess_mat = wrapped_methods["guess_mat"]
        make_func = wrapped_methods["target_type_methods"][target_type][0]
        normalize = wrapped_methods["target_type_methods"][target_type][1]
        apply_constraints = wrapped_methods["constraints"]
        write_a_matrix = wrapped_methods["write_a_matrix"]

    update_a = _get_update_fun(
        update_method=method,
        wrapped_methods=wrapped_methods,
        make_func=make_func,
        gn_config=gn_kwargs,
        fit_config=fit_kwargs,
        jac_config=jac_kwargs,
    )
    # last_t = fit_kwargs["last_t"]
    if last_t == 0:
        last_t = len(target)
    target = target[:last_t:t_spacing]
    a_mem = a_mem[:last_t:t_spacing]
    target_norm, renorm = normalize(target.T)

    a = a_mem.T
    target = target.T
    a_norm = a.copy()
    a_norm[0] = a_norm[0] / renorm[0]
    a_norm[1] = a_norm[1] / renorm[1]

    p_guess, dim = _get_initial_guess(
        Amat_file,
        dim,
        normalize_a_matrix,
        renorm,
        p_guess,
        wrapped_methods,
        target_norm,
        guess_mat,
        mat_to_param,
    )

    # make sure, that fit_config is compatible with iomk.
    fit_kwargs["dim"] = dim
    fit_kwargs["function_type"] = function_type
    fit_kwargs["target_type"] = target_type
    fit_kwargs["Amat_file"] = ""  # For fitting, always use generic guess
    fit_kwargs["p_guess"] = []  # For fitting, always use generic guess
    fit_kwargs["t_spacing"] = 1  # Proper slicing already done above
    fit_kwargs["last_t"] = 0  # Proper slicing already done above

    p0 = apply_constraints(p_guess, dim)
    func = make_func(dim)
    func_w = _wrap_func(func, target_norm[0])

    # Algorithm
    it = start_it

    try:
        os.mkdir(f"it{it}_mem_fit")
    except:
        pass
    cur = func_w(p0)
    np.savetxt(
        f"it{it}_mem_fit/fit", np.array([target_norm[0] * renorm[0], cur * renorm[1]]).T
    )
    write_a_matrix(p0, f"it{it}_mem_fit/Amatrix", renorm=renorm)

    av_sq_res = np.inf
    while av_sq_res > tol_r and it <= max_it + start_it:
        # create directory for step
        os.chdir(dir_path)
        stp_dir = r"it" + str(it) + "/"
        try:
            os.mkdir(stp_dir)
        except:
            pass
        os.chdir(stp_dir)
        for sim_file in sim_files:
            shutil.copy2(
                dir_path + sim_file, dir_path + stp_dir + sim_file.split("/")[-1]
            )
        shutil.copy2(
            dir_path + f"it{it}_mem_fit/Amatrix", dir_path + stp_dir + f"Amatrix"
        )
        print(f"Running LAMMPS command '{lmp}' in folder {stp_dir}.")
        sim_v = subprocess.call(lmp, shell=True)
        if sim_v != 0:
            raise RuntimeError("lammps simulation crashed")
        print(f"LAMMPS finished in folder {stp_dir}.")

        vacf = eval_current_vacf(traj_name, n_traj)
        if vacf[-1, 0] < target[0][-1]:
            raise RuntimeError(
                f"Attempting to match memory up to {target[0][-1]}, but VACF sampled only up to {vacf[-1,0]} time units. Trajctory too short?"
            )
        if (
            target[0][1] % vacf[1, 0] > 0.0001
            and (target[0][1] % vacf[1, 0]) / vacf[1, 0] < 0.9999
        ):
            raise ValueError("Targets timestep is not a multiple of VACF time step.")

        G_gle = mem_from_vacf(
            vacf,
            round(target_norm[0][1] / vacf[1, 0] * renorm[0]) * len(target_norm[0]) * 2,
            round(target_norm[0][1] / vacf[1, 0] * renorm[0]),
        )[: len(target_norm[0])]
        # TODO tracking errors
        # r = G_gle/renorm[0] - target_norm[1]
        r = G_gle / renorm[1] - target_norm[1]
        av_sq_res = np.average(r**2)
        res_file = open("res.txt", "w")
        res_file.write(f"{it}  {av_sq_res}\n")
        res_file.close()
        if remove_traj == True:
            print("Removing trajectory files.")
            for _traj in glob.glob("*.h5"):
                os.remove(_traj)

        # Carry out the actual update
        p0 = update_a(target, a, G_gle, renorm, dir_path, dim, p0, it)
        it += 1
    if not (av_sq_res > tol_r):
        print(f"Tolerance of {tol_r} REACHED!\n")
    else:
        print(f"Tolerance of {tol_r} NOT reached!\n")
    final_it = it - 1
    os.chdir(dir_path)
    return (final_it,)


def add_to_kwargs(old_kwargs, new_kwargs):
    if new_kwargs:
        for k, v in new_kwargs.items():
            if not isinstance(v, dict):
                old_kwargs.update({k: v})


def _get_update_fun(
    update_method, wrapped_methods, make_func, gn_config, fit_config, jac_config
):
    if update_method == "GN":
        apply_constraints = wrapped_methods["constraints"]
        write_a_matrix = wrapped_methods["write_a_matrix"]

        def update_a(target, a, G_gle, renorm, dir_path, dim, p0, it):
            func = make_func(dim)
            func_w = _wrap_func(func, target[0] / renorm[0])
            return _update_a_gn(
                target,
                a,
                G_gle,
                renorm,
                func_w,
                func,
                dir_path,
                p0,
                it,
                dim=dim,
                write_amatrix=write_a_matrix,
                apply_constraints=apply_constraints,
                jac_kwargs=jac_config,
                **gn_config,
            )

        return update_a
    elif update_method == "N":
        apply_constraints = wrapped_methods["constraints"]
        write_a_matrix = wrapped_methods["write_a_matrix"]

        ## TODO This block should be made cleaner
        def add_to_kwargs(fit_kwargs, kwargs):
            if kwargs:
                for k, v in kwargs.items():
                    if not isinstance(v, dict):
                        fit_kwargs.update({k: v})

        fit_kwargs = {}
        add_to_kwargs(fit_kwargs, fit_config)
        add_to_kwargs(fit_kwargs, fit_config.get("GN"))
        add_to_kwargs(fit_kwargs, fit_config.get("GN").get("kwargs"))
        add_to_kwargs(fit_kwargs, fit_config.get("scipy"))
        add_to_kwargs(fit_kwargs, fit_config.get("scipy").get("kwargs"))
        ##############################################################

        def update_a(target, a, G_gle, renorm, dir_path, dim, p0, it):
            return _update_a_n(
                target,
                a,
                G_gle,
                dir_path,
                p0,
                it,
                dim=dim,
                jac_kwargs=jac_config,
                **fit_kwargs,
            )

        return update_a
    else:
        raise ValueError("Illegal update method.")


def _update_a_n(
    target,
    a,
    G_gle,
    dir_path,
    p0,
    it,
    dim=11,
    reg_fun="",
    reg_param=1,
    jac_kwargs=iomk_lib.defaults["jac"],
    **kwargs,
):
    """Update Amatrix in IOMK method. Relies on fitting the updated memory kernel.

    Returns:
        list: Parameters for next iteration.
    """
    reg_fun = _reg_methods[reg_fun](reg_param)
    cur = np.loadtxt(dir_path + f"it{it}_mem_fit/fit")[: len(target[0]), 1]
    it = it + 1
    _iomk_update(target.T, a.T, G_gle, cur, dir_path + f"it{it}_mem")
    fit(
        dir_path + f"it{it}_mem",
        dim=dim,
        reg_fun=reg_fun,
        **jac_kwargs,
        **kwargs,
    )

    return p0.copy()


def _update_a_gn(
    target,
    a,
    G_gle,
    renorm,
    func_w,
    func,
    dir_path,
    p0,
    it,
    # t_vacf_f="",
    dim=11,
    write_a_matrix=write_a_matrix_full,
    reg_fun="",
    reg_param=1,
    apply_constraints=_apply_mat_constraints,
    opt_reg=False,
    # rel_err=False,
    scale_reg=[0.01, 1.0, 100],
    jac_kwargs=iomk_lib.defaults["jac"],
    **kwargs,
):
    """Update Amatrix in IOMK-GN method.

    Returns:
        list: Parameters for next iteration.
    """
    G_gle_norm = G_gle / renorm[1]
    target_norm = [target[0] / renorm[0], target[1] / renorm[1]]
    a_norm = [a[0] / renorm[0], a[1] / renorm[1]]
    reg_fun = _reg_methods[reg_fun](reg_param)
    cur_norm = func_w(p0)

    # Prepare vectors and matrices for Gauss-Newton step
    J = _get_jacobian(func_w, p0, G_gle_norm, a_norm, jac_kwargs)
    r = G_gle_norm - target_norm[1]
    b = np.matmul(J.transpose(), r)
    A = np.matmul(J.transpose(), J)

    # For regularization any function with the same function header can be defined and provided.
    # Should return either a scalar or a diagonal dim x dim matrix
    reg_param = reg_fun(b, p0)
    A = A + reg_param * np.identity(len(p0))
    delta = np.linalg.solve(A, b)
    param_new = apply_constraints(p0, dim, delta)

    # Optimize regularization parameters
    if opt_reg:
        if not isinstance(scale_reg, list):
            raise ValueError("scale_reg has to be a list of length 3.")
        elif not len(scale_reg):
            raise ValueError("scale_reg has to be a list of length 3.")
        else:
            scales = np.linspace(scale_reg[0], scale_reg[1], num=scale_reg[2])
        A0 = np.matmul(J.transpose(), J)
        rsq = np.average((func_w(param_new) - target_norm[1]) ** 2)
        rsq_list = []
        param_list = []
        for scale in scales:
            A = A0 + scale * reg_param * np.identity(len(p0))

            delta = np.linalg.solve(A, b)
            param_new = apply_constraints(p0, dim, delta)

            rsq = (
                a_norm[1][1:]
                + ((G_gle_norm[1:] - a_norm[1][1:]) / cur_norm[1:])
                * func_w(param_new)[1:]
                - target_norm[1][1:]
            ) ** 2

            rsq = np.average(rsq)

            rsq_list.append(rsq)
            param_list.append(param_new.copy())

        min_index = np.argmin(rsq_list)
        param_new = param_list[min_index]

        print(f"scaling with {scales[min_index]}")

    p0 = param_new.copy()
    it += 1
    os.chdir(dir_path)
    try:
        os.mkdir(f"it{it}_mem_fit")
    except:
        pass

    cur_out = np.zeros((2, 2 * len(target_norm[0])))
    dt = target_norm[0][1]
    cur_out[0] = np.arange(len(cur_out[0])) * dt
    cur_out[1] = func(cur_out[0], p0)
    np.savetxt(
        f"it{it}_mem_fit/fit",
        np.array([cur_out[0] * renorm[0], cur_out[1] * renorm[1]]).T,
    )
    write_a_matrix(p0, f"it{it}_mem_fit/Amatrix", renorm=renorm)
    return p0.copy()


def _iomk_update(target, a, G_gle, cur, out_file):
    G_tilde = target.copy()
    G_tilde[1:, 1] = (target[1:, 1] - a[1:, 1]) / (G_gle[1:] - a[1:, 1]) * cur[1:]

    np.savetxt(out_file, G_tilde)


def eval_current_vacf(traj_name, n_traj):
    if n_traj > 1:
        for i in range(1, n_traj + 1):
            print(i)
            get_vacf(traj_name + str(i) + ".h5", str(i) + "_vacf")
        # continue
        temp = np.loadtxt("1_vacf")

        vacf = np.zeros((len(temp), n_traj))
        for i in range(n_traj):
            vacf[:, i] = np.loadtxt(str(i + 1) + "_vacf")[:, 1]

        out = np.zeros((len(temp), 3))
        out[:, 0] = temp[:, 0]
        out[:, 1] = np.average(vacf, axis=1)
        out[:, 2] = np.std(vacf, axis=1, ddof=1)
        np.savetxt("av_vacf", out)
    else:
        get_vacf(traj_name + ".h5", "vacf")
        out = np.loadtxt("vacf")
    return out[:, :2]


@njit
def get_vacf_jac(target_norm, renorm, G_gle_norm, vacf, J_in):
    J = np.zeros_like(J_in)
    for i in range(1, len(J)):
        for n in range(len(J[0])):
            temp = 0.5 * (
                J_in[i][n] * renorm[1] * vacf[0] + G_gle_norm[i] * renorm[1] * J[0][n]
            )
            for j in range(1, i):
                temp = temp + (
                    J_in[i - j][n] * renorm[1] * vacf[j]
                    + G_gle_norm[i - j] * renorm[1] * J[j][n]
                )
            J[i][n] = -target_norm[0][1] * renorm[0] * temp
    return J


@njit
def get_approx_vacf(target_norm, renorm, G_norm, t_vacf):
    """_summary_

    Args:
        target_norm (_type_): _description_
        renorm (_type_): _description_
        G_norm (_type_): _description_
        t_vacf (_type_): _description_

    Returns:
        _type_: _description_
    """
    vacf = np.zeros_like(t_vacf)
    vacf[0] = t_vacf[0]
    for i in range(1, len(vacf)):
        temp = 0.5 * G_norm[i] * renorm[1] * vacf[0]
        for j in range(1, i):
            temp = temp + G_norm[i - j] * vacf[j]
        vacf[i] = vacf[0] - target_norm[0][1] * renorm[0] * temp

    return vacf


def _get_initial_guess(
    Amat_file,
    dim,
    normalize_a_matrix,
    renorm,
    p_guess,
    wrapped_methods,
    target_norm,
    guess_mat,
    mat_to_param,
):
    if Amat_file:
        Amat = np.loadtxt(Amat_file)
        dim = len(Amat)

        print("A-matrix file given. Overwriting dim and initial guess")

        Amat = normalize_a_matrix(Amat, renorm)
        p_guess = mat_to_param(Amat)

    elif len(p_guess) != 0:
        dim = wrapped_methods["get_dim"](len(p_guess))
    else:
        dim = dim
        p_guess = guess_mat(dim, len(target_norm[0]) - 1)
    if len(p_guess) == 0:
        raise RuntimeError(f"Please provide meaningful initial guess or dim parameter.")
    return p_guess, dim


def _get_jacobian(func_w, p0, G_gle, a, jac_kwargs):
    J = approx_derivative(func_w, p0, **jac_kwargs)

    fun_v = func_w(p0)

    for i in range(1, len(a[1])):
        for j in range(len(J[i])):
            J[i][j] = (G_gle[i] - a[1][i]) / fun_v[i] * J[i][j]
    return J
