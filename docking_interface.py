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
import re
import sys

from Bio import PDB
from pymol import cmd
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


def update_config(config_path, cluster, pdb_dir):
    """
    Load the configuration file and update it with the chains sizes.

    :param config_path: the path to the configuration file.
    :type config_path: str
    :param cluster: the cluster ID.
    :type cluster: str
    :param pdb_dir: the path to the clusters PDB directory.
    :type pdb_dir: str
    :return: the updated configuration file.
    :rtype: dict
    """
    # load the config file
    try:
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file.read())
    except ImportError as exc:
        logging.error(exc, exc_info=True)
    # get the chains sizes and update the config file
    parser_pdb = PDB.PDBParser()
    structure = parser_pdb.get_structure(cluster_id, os.path.join(pdb_dir, "{}.pdb".format(cluster)))
    model = structure[0]
    for chain in model:
        config["chain {}".format(chain.id)]["length"] = len(list(chain.get_residues()))

    return config


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


def get_interface(cluster, input_dir, prefix, config):
    """
    Get the interface residues between chain A and chain B.

    :param cluster: the cluster ID from HADDOCK outputs.
    :type cluster: str
    :param input_dir: the path to the HADDOCK outputs directory.
    :type input_dir: str
    :param prefix: the prefix of the outputs files.
    :type prefix: str
    :param config: the configuration of the analysis.
    :type config: dict
    :return:
    :rtype:
    """

    cmd.do("load {}".format(os.path.join(input_dir, "{}.pdb".format(cluster))))
    cmd.show("cartoon", cluster)
    for chain in config:
        # set chain color
        cmd.color(config[chain]["color"], chain)
        logging.debug("{} ({}): color is set to {}.".format(chain, config[chain]["name"], config[chain]["color"]))
        # set regions if any
        if "regions" in config[chain]:
            for region_id, region_data in config[chain]["regions"].items():
                cmd.select(region_id, "{} and resi {}-{}".format(chain, region_data["start"], region_data["end"]))
                cmd.color(region_data["color"], region_id)
                logging.debug("\tregion {}: from {} to {}, color is set to {}.".format(region_id,
                                                                                       region_data["start"],
                                                                                       region_data["end"],
                                                                                       region_data["color"]))
        # if "mutations" in config:
        #     if "alterations" in config["mutations"]:


    # path_img = os.path.join(out_dir, "{}.png".format(cluster))
    # cmd.png(path_img, ray=1)


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
    parser.add_argument("-p", "--prefix", required=True, type=str, help="the prefix path for the output files.")
    parser.add_argument("-c", "--config", required=True, type=str, help="the path to the YAML configuration file.")
    parser.add_argument("-l", "--log", required=False, type=str,
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
    out_dir = os.path.dirname(args.prefix)
    os.makedirs(out_dir, exist_ok=True)

    # create the logger
    if args.log:
        log_path = args.log
    else:
        log_path = os.path.join(out_dir, "{}.log".format(os.path.splitext(os.path.basename(__file__))[0]))
    create_log(log_path, args.log_level)

    logging.info("version: {}".format(__version__))
    logging.info("CMD: {}".format(" ".join(sys.argv)))

    # get the pdb cluster file with the minimal binding energy
    cluster_id = get_cluster_min_binding_energy(args.input)

    # load the configuration file and update it
    configuration = update_config(args.config, cluster_id, args.input)

    # load the pdb file and get the interface residues
    get_interface(cluster_id, args.input, args.prefix, configuration)


