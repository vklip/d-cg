import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize import curve_fit

import iomk_lib
from iomk_lib._fit_tools import _apply_mat_constraints, _reg_methods
from iomk_lib._tools import _get_dim_full

__all__ = [
    "_fit_GN",
    "_fit_scipy",
    "_fit_types",
]
""" Main fitting functions """

# Use Gauss-Newton scheme to fit K or G. Should (normally) not be run directly! Wrapper takes care of proper input!
# If really needed, make sure to provide proper initial guess, make sure that dim is consistent with parameters and function type
# normalization and renormalization is also taken care of elsewhere


def _fit_GN(
    funcs,
    arff,
    p_guess,
    apply_constraints=_apply_mat_constraints,
    get_a_dim=_get_dim_full,
    reg_fun="adaptive",  # Directly providing function also works. E.g. reg_adaptive(1)
    reg_param=1,
    tol_r=1e-7,
    max_steps=100,
    output_folder="",
    jac_kwargs=iomk_lib.defaults["jac"],
    **kwargs,
):
    func_w = funcs[0]
    if isinstance(reg_fun, str):
        reg_fun = _reg_methods[reg_fun](reg_param)
    else:
        pass
    p0 = p_guess.copy()
    dim = get_a_dim(len(p0))
    p0 = apply_constraints(p0, dim)
    r = func_w(p0) - arff[1]
    # just outsourced to not be reallocated in every loop
    I = np.identity(len(p0))

    # initializing log data
    av_sq_res = np.average(r**2)
    res_list = np.ones(max_steps + 1) * 100000
    param_list = np.zeros((max_steps + 1, len(p0)))

    count = 0
    res_list[count] = av_sq_res
    param_list[count, :] = p0
    not_converged = True
    if output_folder:
        if isinstance(output_folder, str):
            res_file = open(output_folder + "/" + "res.txt", "w")
        else:
            raise ValueError()
    print(f"step: {count} <r^2>:  + {av_sq_res}")
    res_file.write(f"# step <r^2>\n")
    res_file.write(f"{count}  {av_sq_res}\n")
    # Converged when average r**2 smaller then tolerance or max_steps reached.
    # Additional criteria might be useful and could be implemented in a later point in time.
    while av_sq_res > tol_r and count < max_steps:
        # Provide parameters for approx_derivative.
        # Predefined values in this function should be fine for normalized problems and descent constraints
        J = approx_derivative(func_w, p0, **jac_kwargs)
        b = np.matmul(J.transpose(), r)
        A = np.matmul(J.transpose(), J)

        # Any function with the same function header can be defined and provided.
        # Should return either a scalar or a diagonal dim x dim matrix
        reg_param = reg_fun(b, p0)
        A = A + reg_param * I

        delta = np.linalg.solve(A, b)

        # Here one could do a line search, but step size is indirectly controlled via reg_param and
        # can be directly optimized via reg_fun(...) and is thus removed for now
        p0 = apply_constraints(p0, dim, delta)

        r = func_w(p0) - arff[1]
        av_sq_res = np.average(r**2)

        count += 1
        res_file.write(f"{count}  {av_sq_res}\n")
        res_list[count] = av_sq_res

        param_list[count, :] = p0
        print("step:" + str(count) + " <r^2>: " + str(av_sq_res))

    print("\n\n---------SUMMARY------------\n\n")
    res_file.close()
    if not (count < max_steps):
        print(
            f"Reached maxmimum number of {max_steps} iterations. \nNOT CONVERGED ACCORDING TO GIVEN SPECIFIED TOLERANCES!\n"
        )
    else:
        print(f"CONVERGED AFTER {count} ITERATIONS!\n")

    if not (av_sq_res > tol_r):
        print(f"Tolerance of {tol_r} REACHED!\n")
    else:
        print(f"Tolerance of {tol_r} NOT reached!\n")

    param_list = param_list[: count + 1]
    res_list = res_list[: count + 1]
    min_i = np.argmin(res_list)

    if count == min_i:
        print(f"Final iteration has minimal error of {res_list[count]}.\n")
    else:
        print(
            f"Minimal error of {res_list[min_i]} found in iteration {min_i}.\nLast iteration has a {(res_list[count]-res_list[min_i])/res_list[min_i]*100} % larger error!\n"
        )
    # Outputs a lot of data.
    # Wrapper really only works with p0 and p_guess for now, but forwards it to user, just in case.
    return [p0, param_list, res_list]


def _fit_scipy(
    funcs,
    arff,
    p_guess,
    tol_r=1e-5,
    max_steps=10000,
    apply_constraints=_apply_mat_constraints,
    get_a_dim=_get_dim_full,
    jac_kwargs=iomk_lib.defaults["jac"],
    output_folder="",
    **kwargs,
):
    """Wraps scipy curve_fit to harmonize in and output with _fit_GN.
    Should (normally) not be run directly! Wrapper takes care of proper input!

       Args:
           funcs (_type_): _description_
           arff (_type_): _description_
           p_guess (list, optional): _description_. Defaults to [].
           tol_r (_type_, optional): _description_. Defaults to 1e-5.
           max_steps (int, optional): _description_. Defaults to 10000.
           apply_constraints (_type_, optional): _description_. Defaults to _apply_mat_constraints.
           get_a_dim (_type_, optional): _description_. Defaults to _get_dim_full.
           jac_method (str, optional): _description_. Defaults to "2-point".
           jac_bounds (tuple, optional): _description_. Defaults to (1e-15, np.inf).
           jac_rel_step (_type_, optional): _description_. Defaults to None.
           jac_abs_step (_type_, optional): _description_. Defaults to 1e-5.

       Returns:
           _type_: _description_
    """
    func_w = funcs[0]
    func = funcs[1]
    p0 = p_guess.copy()
    dim = get_a_dim(len(p0))

    lbounds = apply_constraints(-np.ones(len(p_guess)) * 1e20, dim)
    ubounds = np.array([np.inf] * len(p_guess))
    bounds = [lbounds, ubounds]

    if output_folder:
        if isinstance(output_folder, str):
            res_file = open(output_folder + "/" + "res.txt", "w")
            res_file.close()
        else:
            raise ValueError()

    global count
    global temp_params
    global param_list
    count = 0
    res_list = np.ones(max_steps + 1) * 1000
    param_list = np.zeros((max_steps + 1, len(p0)))

    def approx_jacobian(x, *args):
        global count
        global temp_params
        r = func_w(args) - arff[1]
        av_sq_res = np.average(r**2)

        # Here we use the callable Jacobian to track the average error.
        temp_params = list(args)
        param_list[count, :] = np.array(temp_params)
        res_file = open(output_folder + "/" + "res.txt", "a")
        res_file.write(f"{count}  {av_sq_res}\n")
        res_file.close()

        print("step:" + str(count) + " err: " + str(av_sq_res))
        if av_sq_res < tol_r:
            raise RuntimeError(
                f"Tolerance of {tol_r} REACHED!\n"
            )  # Handle manual convergence criterion within curve_fit
        count += 1
        J = approx_derivative(
            func_w,
            args,
            **jac_kwargs,
        )

        return J  # approx_derivative(func, x, method="2-point", abs_step=epsilon, args=args)

    try:
        params, pcov = curve_fit(
            lambda x, *params: func(x, params),
            arff[0],
            arff[1],
            p0=p_guess,
            maxfev=max_steps
            + 2,  # add 2 as otherwise the final residues and parameters are not written if scipy
            bounds=bounds,
            full_output=False,
            jac=approx_jacobian,
        )
    except RuntimeError as e:
        print(e)
        params = temp_params.copy()
    return [params, param_list, res_list]


_fit_types = {"scipy": _fit_scipy, "GN": _fit_GN}
