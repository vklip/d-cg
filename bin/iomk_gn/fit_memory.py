#!/usr/bin/env python

import argparse
import tomlkit
import numpy as np
import os

import iomk_lib
import iomk_lib.fit as fit


# TODO up to now full merge. Should warn about unknown parameters.
def merge_config(a, b):
    for k in b:
        if k in a and isinstance(a[k], dict) and isinstance(b[k], dict):
            merge_config(a[k], b[k])
        else:
            a[k] = b[k]


def add_to_kwargs(fit_kwargs, kwargs):
    if kwargs:
        for k, v in kwargs.items():
            if not isinstance(v, dict):
                fit_kwargs.update({k: v})


def main():
    parser = argparse.ArgumentParser(
        description="Fit a memory kernel to extract drift matrix for auxiliary variable thermostat"
    )

    parser.add_argument(
        "-t",
        metavar="file",
        type=str,
        help="/path/to/memory_kernel",
    )
    parser.add_argument(
        "-c", metavar="config", type=str, help="config file in toml format"
    )
    parser.add_argument("-o", metavar="output folder", type=str, help="output folder")

    parser.add_argument(
        "--log_config",
        metavar="logging config on/off",
        help="Turn on/off writing of full configuration",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--log",
        metavar="logging results on/off",
        help="Turn on/off writing of full configuration",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--norm",
        metavar="write normalized data on/off",
        help="Turn on/off normalized target, guess and fit",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--min",
        metavar="write data with minimal residues on/off",
        help="Turn on/off writing of Amatrx, fit and normalized fit with minimal residues",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args = parser.parse_args()
    _main(args)


# This allows to call the main function from another python script. Parameters are now wrapped in a namespace.
# Just collect all command line arguments in a dict ("args_dict") and call "_main(argparse.Namespace(**args_dict))"
def _main(args):
    config = tomlkit.loads(tomlkit.dumps(iomk_lib.defaults))
    if args.c:
        config_f = open(args.c)
        merge_config(config, tomlkit.load(config_f))
        config_f.close()

    if args.t:
        config["general"]["target_file"] = args.t

    if args.o:
        config["fit"]["output_folder"] = args.o

    if not config["general"]["target_file"]:
        print(
            "Please provide a file name for the target (via -t within the command line or in a configuration file provided via -c). \n"
            + f"{os.path.basename(__file__)} -h to show more options."
        )
        exit()
    else:
        try:
            temp = open(config["general"]["target_file"])
            temp.close()
        except:
            print(f"Could not open the target file: {config['general']['target_file']}")
            exit()

    # Unwrap configuration to kwargs
    fit_kwargs = {}
    add_to_kwargs(fit_kwargs, config.get("general"))
    add_to_kwargs(fit_kwargs, config.get("jac"))
    add_to_kwargs(fit_kwargs, config.get("fit"))
    add_to_kwargs(fit_kwargs, config.get("fit").get("GN"))
    add_to_kwargs(fit_kwargs, config.get("fit").get("GN").get("kwargs"))
    add_to_kwargs(fit_kwargs, config.get("fit").get("scipy"))
    add_to_kwargs(fit_kwargs, config.get("fit").get("scipy").get("kwargs"))
    results, target_norm, func, write_a_matrix, get_dim = fit.fit(**fit_kwargs)

    # unwrapping all output data to handle writing
    renorm = target_norm[1]
    target_norm = target_norm[0]
    p_final = results[0]
    param_list = results[1]
    res_list = results[2]
    func_w = func[1]
    func = func[0]
    output_folder = config.get("fit").get("output_folder")
    if not output_folder:
        output_folder = config.get("general").get("target_file") + "_fit"

    if args.min:
        fit_out = np.zeros((2, 3 * len(target_norm[0])))
        dt = target_norm[0][1]
        fit_out[0] = np.arange(len(fit_out[0])) * dt
        min_i = np.argmin(res_list)
        p_min = param_list[min_i, :]
        fit_out[1] = func(fit_out[0], p_min)
        fit_out[0] *= renorm[0]
        fit_out[1] *= renorm[1]
        if config.get("fit").get("target_type") == "K":
            fit_out[1] /= renorm[0]
        np.savetxt(f"{output_folder}/fit_min", fit_out.T)
        write_a_matrix(p_min, output_folder + "/Amatrix", renorm=renorm)

    if args.norm:
        np.savetxt(f"{output_folder}/renorm", renorm)
        np.savetxt(f"{output_folder}/target_norm", np.array(target_norm).T)

        p_guess = results[1][0]
        np.savetxt(
            f"{output_folder}/fit_start_norm",
            np.array([target_norm[0], func_w(p_guess)]).T,
        )
        p_final = results[0]
        np.savetxt(
            f"{output_folder}/fit_norm", np.array([target_norm[0], func_w(p_final)]).T
        )
        min_i = np.argmin(res_list)
        p_min = param_list[min_i, :]
        np.savetxt(
            f"{output_folder}/fit_min_norm",
            np.array([target_norm[0], func_w(p_min)]).T,
        )

    if args.log_config:
        log_config_file = "log_config.toml"
        log_config_file = f"{output_folder}/{log_config_file}"
        log = open(log_config_file, "w")
        tomlkit.dump(config, log)
        log.close()
    if args.log:
        logfile = "fit.log"
        logfile = f"{output_folder}/{logfile}"
        log = open(logfile, "w")
        log.write(f"Summary\n\n")
        log.write(
            f"Full target file path: {os.path.abspath(config.get('general').get('target_file'))}\n\n"
        )
        log.write(f"Drift matrix dimension: {get_dim(len(p_final))}\n")
        log.write(f"Drift matrix function: {config.get('fit').get('function_type')}\n")
        log.write(f"Target type: {config.get('fit').get('target_type')}\n")
        log.write(f"Fitting method: {config.get('fit').get('fit_type')}\n")
        if args.log_config:
            log.write(
                f"Full configuration written to: {os.path.abspath(log_config_file)}\n"
            )
        tol_r = config.get("fit").get("tol_r")
        log.write(
            f"Converged to requested tolerance of {tol_r}?: {res_list[len(res_list)-1]<= tol_r}\n\n"
        )
        final_min = len(res_list) - 1 == min_i
        log.write(f"Minimal residue at final iteration?: {final_min}\n\n")
        if not final_min:
            log.write(f"Minimal residue at iteration: {min_i}\n")
        log.write(f"Minimal averaged squared residue value: {res_list[min_i]}\n")
        log.close()


if __name__ == "__main__":
    main()
