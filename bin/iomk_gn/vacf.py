#!/usr/bin/env python

import argparse
import tomlkit


import iomk_lib
from iomk_lib._analyze_trj import readtrj, get_vacf


def main():
    parser = argparse.ArgumentParser(
        description="Extract velocity auto-correlation function from trajectory."
    )

    parser.add_argument(
        "-t",
        metavar="trajectory file",
        type=str,
        help="/path/to/h5md_trajectory_file",
    )
    parser.add_argument(
        "-o",
        metavar="output file ",
        type=str,
        help="/path/to/output_vacf_file",
    )
    args = parser.parse_args()
    get_vacf(args.t, args.o)


if __name__ == "__main__":
    main()
