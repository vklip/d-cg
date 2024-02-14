#!/usr/bin/env python

import argparse
import tomlkit


import iomk_lib
from iomk_lib._analyze_trj import mem_from_vacf


def main():
    parser = argparse.ArgumentParser(
        description="Extract velocity auto-correlation function from trajectory."
    )

    parser.add_argument(
        "-f",
        metavar="vacf file",
        type=str,
        help="/path/to/vacf_file",
    )
    parser.add_argument(
        "-o",
        metavar="output file ",
        type=str,
        help="/path/to/output_imem_file",
    )
    parser.add_argument(
        "-n",
        metavar="output file ",
        type=int,
        help="/path/to/output_imem_file",
    )
    parser.add_argument(
        "-s",
        metavar="output file ",
        type=int,
        help="/path/to/output_imem_file",
    )
    args = parser.parse_args()
    print(iomk_lib._analyze_trj.__file__)
    if not args.o:
        mem_from_vacf(args.f, args.n, args.s)
    else:
        print("HERE")
        print(args.o)
        mem_from_vacf(args.f, args.n, args.s, args.o)


if __name__ == "__main__":
    main()
