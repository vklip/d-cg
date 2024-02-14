__all__ = ["fit"]
import os
import subprocess
import numpy as np

# import
from iomk_lib._fit_tools import _wrapped_methods
from iomk_lib._fit import _fit_types
from iomk_lib._fit_tools import *
from iomk_lib._tools import *


def fit(
    target_file,
    target_type="G",
    fit_type="GN",
    function_type="full_matrix",
    dim=7,
    output_folder="",
    t_spacing=1,
    last_t=0,
    max_steps=100,
    Amat_file="",
    p_guess=[],
    tol_r=1e-7,
    **kwargs,
):
    """Main function to directly extract a drift matrix for the auxiliary
        variable thermostat due to Ceriotti (as implemented in LAMMPS) by fitting of a memory kernel.

        Args:
            target_file (str): path/to/file of target function.
            target_type (str, optional):  How to interpret the target array. Options: {"G": integrated memory kernel (in frequency units), "K":memory kernel (in frequency^2 units) }. Defaults to "G".
            fit_type (str, optional): Which method to use for fitting. Options: {"GN": Gauss-Newton scheme. Additional parameters read from [fit.GN],
    #"scipy": Wrapped use of scipy.optimize.curve_fit with bounds. Further options are forwarded via [fit.scipy.kwargs]}. Defaults to "GN".
            function_type (str, optional): # Defines the functional form of the memory kernel. Options: {"full_matrix" use a full matrix with screw-symmetric off diagonal elements, "dosz" use damped oscillators}. Defaults to "full_matrix".
            dim (int, optional): Fixes Amatrix dimension. Is overwritten if "Amat_file" or "p_guess" is provided. When neither "Amat_file" nor "p_guess" is provided, and initial guess is generically generated with dim dimensionality. Defaults to 7.
            output_folder (str, optional): Write results to this folder. If empty, determined from "target_file" Defaults to "".
            t_spacing (int, optional): Only consider every "t_spacing"th frame from target. Defaults to 1.
            last_t (int, optional): Only fit up to this entry. If 0, considers full input. Defaults to 0.
            max_steps (int, optional): Maximum number of optimization steps. Defaults to 100.
            Amat_file (str, optional): path/to/Amatrix. Used as initial guess. Gets overwritten, when p_guess is provided. CAUTION: Only provide input consistent with the chosen "function_type". Defaults to "".
            p_guess (list, optional): List of parameters. Overwrites "dim" AND "Amat_file". CAUTION: Currently assumes that parameters are already normalized. In most cases, "dim" or "Amat_file" should be used to generate initial guess.  Defaults to [].
            tol_r (float, optional): Residue tolerance. Optimization stops when mean squared error is below tol_r. Defaults to 1e-7.

        Raises:
            ValueError: Raised when invalid input is provided.


        Returns:
            tupel: Stores information about the fitting process: (results, (target_norm, renorm), (func, func_w), write_a_matrix, dim_method)
                - results: [final parameters, list of parameters for every iteration, residues in every iteration]
                - (target_norm, renorm): Normalized targed function used for fitting, and normalization constants to restore original.
                - (func, func_w): function to evaluate from parameters and time array and wrapped version which only takes parameters for fixed time array.
                - write_a_matrix: method to write Amatrix file for a set of parameters, consistent with definitions used in the optimization process. Can be used to store an Amatrix for any iteration in post-processing.
                - dim_method: Method to calculate the dimensionality of the Amatrix from parameters.

                Most of the returned data is not typically needed when using the method, as most important results are stored in files anyways.
                Data is returned for testing or more thorough in code analysis of the optimization process.
    """

    # Check if input is meaningful and unwrap specific functions
    if fit_type not in _fit_types.keys():
        raise ValueError(f'Unkown fit_type "{fit_type}".')
    else:
        fit_func = _fit_types[fit_type]
    if function_type not in _wrapped_methods.keys():
        raise ValueError(f'Unkown function_type "{function_type}".')
    elif (
        target_type not in _wrapped_methods[function_type]["target_type_methods"].keys()
    ):
        raise ValueError(
            f'Unknown target_type "{target_type}" for function_type "{function_type}".'
        )
    else:
        wrapped_methods = _wrapped_methods[function_type]
        mat_to_param = wrapped_methods["mat_to_param"]
        guess_mat = wrapped_methods["guess_mat"]
        make_func = wrapped_methods["target_type_methods"][target_type][0]
        normalize = wrapped_methods["target_type_methods"][target_type][1]
        dim_method = wrapped_methods["get_dim"]
        apply_constraints = wrapped_methods["constraints"]
        write_a_matrix = wrapped_methods["write_a_matrix"]

    # Read, crop and normalize target
    target = np.loadtxt(target_file)
    if last_t == 0:
        last_t = len(target)
    target_norm, renorm = normalize(target[:last_t:t_spacing].T)

    # Generate initial guess either from user or from a-priori heuristic.
    # Fix dimensionality of drift matrix accordingly.
    if Amat_file:
        # It is assumed, that the given matrix file is NOT normalized.
        Amat = np.loadtxt(Amat_file)
        if not Amat.shape == 2:
            raise ValueError(f"Invalid matrix in {Amat_file}")
        elif not len(Amat[:, 0]) == len(Amat[0, :]):
            raise ValueError(f"Invalid matrix in {Amat_file}")
        dim = len(Amat)
        Amat = normalize_a_matrix(Amat, renorm)
        print("A-matrix file given. Overwriting dim and initial guess")
        p_guess = mat_to_param(Amat)

    elif len(p_guess) != 0:
        # If a list of parameters is directly given,
        # it is assumed that the parameters are supposed to be used as is, without normalization.
        print("Using provided initial guess for parameters.")
        dim = dim_method(len(p_guess))

    else:
        print("No initial guess for parameters provided. Using generic initial guess")
        p_guess = guess_mat(dim, len(target_norm[0]) - 1)

    # Generate function wrapper
    func = make_func(dim)
    func_w = _wrap_func(func, target_norm[0])

    # If no output folder is specified use this
    if not output_folder:
        output_folder = f"{target_file}_fit"

    try:
        os.mkdir(f"{output_folder}")
    except:
        pass

    # Preparation steps finished, run main function
    results = fit_func(
        (func_w, func),
        target_norm,
        p_guess=p_guess,
        apply_constraints=apply_constraints,
        get_a_dim=dim_method,
        output_folder=output_folder,
        max_steps=max_steps,
        tol_r=tol_r,
        **kwargs,
    )

    # Write minimal output: re-normalized matrix, final iteration and initial guess
    write_a_matrix(results[0], output_folder + "/Amatrix", renorm=renorm)
    fit_out = np.zeros((2, 3 * len(target_norm[0])))
    dt = target_norm[0][1]
    fit_out[0] = np.arange(len(fit_out[0])) * dt
    fit_out[1] = func(fit_out[0], results[0])
    fit_out[0] *= renorm[0]
    fit_out[1] *= renorm[1]

    if target_type == "K":
        fit_out[1] /= renorm[0]
    np.savetxt(f"{output_folder}/fit", fit_out.T)
    fit_out = np.zeros((2, 3 * len(target_norm[0])))
    fit_out[0] = np.arange(len(fit_out[0])) * dt
    fit_out[1] = func(fit_out[0], results[1][0])
    fit_out[0] *= renorm[0]
    fit_out[1] *= renorm[1]

    if target_type == "K":
        fit_out[1] /= renorm[0]
    np.savetxt(f"{output_folder}/fit_start", fit_out.T)

    return results, (target_norm, renorm), (func, func_w), write_a_matrix, dim_method
