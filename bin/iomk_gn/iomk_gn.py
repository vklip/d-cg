#!/usr/bin/env python

import argparse
import tomlkit
import os

import iomk_lib
import iomk_lib.iomk_gn


# TODO up to now full merge. Should warn about unknown parameters.
def merge_config(a, b):
    for k in b:
        if k in a and isinstance(a[k], dict) and isinstance(b[k], dict):
            merge_config(a[k], b[k])
        else:
            a[k] = b[k]


def main():
    parser = argparse.ArgumentParser(
        description="Fit a memory kernel to extract drift matrix or auxiliary variable thermostat"
    )

    parser.add_argument(
        "-t",
        metavar="file",
        type=str,
        help="/path/to/memory_kernel",
    )

    parser.add_argument(
        "-a",
        metavar="file",
        type=str,
        help="/path/to/CG-MD_memory_kernel",
    )
    parser.add_argument(
        "-g",
        metavar="file",
        type=str,
        help="/path/to/initial_guess_Amatrix",
    )
    parser.add_argument(
        "-c", metavar="config", type=str, help="config file in toml format"
    )
    parser.add_argument("-o", metavar="output folder", type=str, help="output folder")
    parser.add_argument(
        "-s",
        metavar="Simulation files",
        type=str,
        nargs="+",
        help="Input files copied into the folder LAMMPS is run in.",
    )
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
    parser.add_argument(
        "--defaults",
        metavar="print defaults to sdout",
        help="print defaults to sdout",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    if args.defaults:
        print(tomlkit.dumps(iomk_lib.defaults))
        exit()

    _main(args)


# This allows to call the main function from another python script. Parameters are now wrapped in a namespace.
# Just collect all command line arguments in a dict ("args_dict") and call "_main(argparse.Namespace(**args_dict))"
def _main(args):
    config = tomlkit.loads(tomlkit.dumps(iomk_lib.defaults))
    print(config)
    if args.c:
        config_f = open(args.c)
        merge_config(config, tomlkit.load(config_f))
        config_f.close()

    if args.t:
        config["general"]["target_file"] = args.t
    if args.a:
        config["iomk"]["a_file"] = args.a

    if args.s:
        config["iomk"]["sim_files"] = list(set(config["iomk"]["sim_files"] + args.s))

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

    iomk_lib.iomk_gn.main_iomk_gn(config)
    if args.log:
        logfile = "log_config.toml"
        output_folder = config.get("fit").get("output_folder")
        if output_folder:
            logfile = f"{output_folder}/{logfile}"
        log = open(logfile, "w")
        tomlkit.dump(config, log)
        log.close()


if __name__ == "__main__":
    main()
