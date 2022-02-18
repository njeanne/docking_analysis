#!/usr/bin/env python3

"""
Created on 17 Feb. 2022
"""

__author__ = "Nicolas JEANNE"
__copyright__ = "GNU General Public License"
__email__ = "jeanne.n@chu-toulouse.fr"
__version__ = "1.0.0"

import argparse
import logging
import os
from pymol import cmd
import re
import sys
import yaml


def create_log(path, level):
    """Create the log as a text file and as a stream.

    :param path: the path of the log.
    :type path: str
    :param level: the level og the log.
    :type level: str
    :return: the logging:
    :rtype: logging
    """

    log_level_dict = {"DEBUG": logging.DEBUG,
                      "INFO": logging.INFO,
                      "WARNING": logging.WARNING,
                      "ERROR": logging.ERROR,
                      "CRITICAL": logging.CRITICAL}

    if level is None:
        log_level = log_level_dict["INFO"]
    else:
        log_level = log_level_dict[args.log_level]

    if os.path.exists(path):
        os.remove(path)

    logging.basicConfig(format="%(asctime)s %(levelname)s:\t%(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S",
                        level=log_level,
                        handlers=[logging.FileHandler(path), logging.StreamHandler()])
    return logging


def get_cluster_min_binding_energy(dir_pdb_cluster):
    """
    Get the ID of the cluster with the minimum of binding energy.

    :param dir_pdb_cluster: the path to the pdb file directory.
    :type dir_pdb_cluster: str
    :return: the ID of the cluster with the minimal binding energy.
    :rtype: str
    """
    pattern_binding_energy = re.compile("Binding energy: (-?\\d+.?\\d+)")
    cluster_min_energy = None
    min_energy = 0.0
    for cluster_file_id in os.listdir(args.input):
        file_split = os.path.splitext(cluster_file_id)
        if file_split[1] == ".pdb":
            with open(os.path.join(dir_pdb_cluster, cluster_file_id)) as cluster_file:
                for line in cluster_file.readlines():
                    match = pattern_binding_energy.search(line)
                    if match:
                        if float(match.group(1)) < min_energy:
                            min_energy = float(match.group(1))
                            cluster_min_energy = file_split[0]
                        continue
    logging.info("{} detected with the minimal binding energy: {}".format(cluster_min_energy, min_energy))
    return cluster_min_energy


def get_interface(cluster, input_dir, out_dir, chain_a, chain_b, mutations_a_path, mutations_b_path, regions_a_path,
                  regions_b_path):
    """
    Get the interface residues between chain A and chain B.

    :param cluster: the cluster ID from HADDOCK outputs.
    :type cluster: str
    :param input_dir: the path to the HADDOCK outputs directory.
    :type input_dir: str
    :param out_dir: the path to the outputs directory.
    :type out_dir: str
    :param chain_a: the name of the chain A for the results.
    :type chain_a: str
    :param chain_b: the name of the chain B for the results.
    :type chain_b: str
    :param mutations_a_path: the path to the file containing the mutations to highlight in the chain A.
    :type mutations_a_path: str
    :param mutations_b_path: the path to the file containing the mutations to highlight in the chain B.
    :type mutations_b_path: str
    :param regions_a_path: the path to the CSV file containing the regions to highlight in the chain A.
    :type regions_a_path: str
    :param regions_b_path: the path to the CSV file containing the regions to highlight in the chain B.
    :type regions_b_path: str
    :return:
    :rtype:
    """
    cmd.do("load {}".format(os.path.join(input_dir, "{}.pdb".format(cluster))))
    cmd.show("cartoon", cluster)
    cmd.color("chocolate", "chain A")
    cmd.color("marine", "chain B")

    path_img = os.path.join(out_dir, "{}.png".format(cluster))
    cmd.png(path_img, ray=1)


if __name__ == "__main__":
    descr = """
    {} v. {}

    Created by {}.
    Contact: {}
    {}

    Distributed on an "AS IS" basis without warranties or conditions of any kind, either express or implied.

    From a docking performed by HADDOCK, select the better cluster and extract the interface atoms.
    """.format(os.path.basename(__file__), __version__, __author__, __email__, __copyright__)
    parser = argparse.ArgumentParser(description=descr, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=str, help="the path to the output directory.")
    parser.add_argument("--chain-A", required=True, type=str,
                        help="the name of the protein as chain A in the docking pdb files.")
    parser.add_argument("--chain-B", required=True, type=str,
                        help="the name of the protein as chain B in the docking pdb files.")
    parser.add_argument("--regions-A", required=False, type=str,
                        help="the path to a CSV with the regions, start position, end position and color to highligth "
                             "in chain A.")
    parser.add_argument("--regions-B", required=False, type=str,
                        help="the path to a CSV with the regions, start position, end position and color to highligth "
                             "in chain B.")
    parser.add_argument("--mutations-A", required=False, type=str,
                        help="the path to a file with the mutations in chain A to highlight. The file contains the "
                             "positions separated by commas.")
    parser.add_argument("--mutations-B", required=False, type=str,
                        help="the path to a file with the mutations in chain A to highlight. The file contains the "
                             "positions separated by commas.")
    parser.add_argument("--log", required=False, type=str,
                        help="the path for the log file. If this option is skipped, the log file is created in the "
                             "output directory.")
    parser.add_argument("--log-level", required=False, type=str,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="set the log level. If the option is skipped, log level is INFO.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("input", type=str,
                        help="the path to the HADDOCK output directory directory.")
    args = parser.parse_args()

    # create output directory if necessary
    os.makedirs(args.out, exist_ok=True)

    # create the logger
    if args.log:
        log_path = args.log
    else:
        log_path = os.path.join(args.out, "{}.log".format(os.path.splitext(os.path.basename(__file__))[0]))
    create_log(log_path, args.log_level)

    logging.info("version: {}".format(__version__))
    logging.info("CMD: {}".format(" ".join(sys.argv)))

    # get the pdb cluster file with the minimal binding energy
    cluster_id = get_cluster_min_binding_energy(args.input)

    # load the pdb file and get the interface residues
    get_interface(cluster_id, args.input, args.out, args.chain_A, args.chain_B, args.mutations_A, args.mutations_B,
                  args.regions_A, args.regions_B)


