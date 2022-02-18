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

sys.path.insert(0, "references")
import interfaceResidues


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
    parser_pdb = PDB.PDBParser(QUIET=True)
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


def get_interfaces(cluster, input_dir, prefix, config):
    """
    Get the interfaces residues between the chains in the docking PDF.

    :param cluster: the cluster ID from HADDOCK outputs.
    :type cluster: str
    :param input_dir: the path to the HADDOCK outputs directory.
    :type input_dir: str
    :param prefix: the prefix of the outputs files.
    :type prefix: str
    :param config: the configuration of the analysis.
    :type config: dict
    :return: the interfaces.
    :rtype: dict
    """

    cmd.do("load {}".format(os.path.join(input_dir, "{}.pdb".format(cluster))))
    cmd.show("cartoon", cluster)
    for chain in config:
        # set chain color
        cmd.color(config[chain]["color"], chain)
        logging.debug("{} ({}): color is set to {}.".format(chain, config[chain]["name"], config[chain]["color"]))
        # set regions and colors if any
        if "regions" in config[chain]:
            for region_id, region_data in config[chain]["regions"].items():
                cmd.select(region_id, "{} and resi {}-{}".format(chain, region_data["start"], region_data["end"]))
                cmd.color(region_data["color"], region_id)
                logging.info("\tregion {}: from {} to {}, color is set to {}.".format(region_id,
                                                                                      region_data["start"],
                                                                                      region_data["end"],
                                                                                      region_data["color"]))
        if "mutations" in config[chain]:
            # update index with alterations from the reference sequence where the mutations index come from
            if "alterations" in config[chain]["mutations"]:
                shift_idx = 0
                for alter_pos, alter_value in config[chain]["mutations"]["alterations"].items():
                    alter_str = "{}{}".format("+" if alter_value > 0 else "-", alter_value)
                    cmd.alter("{} and resi {}-{}".format(chain, alter_pos + shift_idx,
                                                         config[chain]["length"] + shift_idx),
                              "resi=str(int(resi){})".format(alter_str))
                    logging.info("\talteration of {} from {} (original position {}).".format(alter_str,
                                                                                             alter_pos + shift_idx,
                                                                                             alter_pos))
                    shift_idx = shift_idx + int(alter_str)
            # set mutations to licorice representation
            mutations_selection = "{}_mutations".format(config[chain]["name"])
            mut_positions_str = config[chain]["mutations"]["positions"][0]
            for idx in range(1, len(config[chain]["mutations"]["positions"])):
                mut_positions_str = "{}+{}".format(mut_positions_str, config[chain]["mutations"]["positions"][idx])
            cmd.select(mutations_selection, "{} and resi {}".format(chain, mut_positions_str))
            cmd.show("licorice", mutations_selection)
            pymol.util.cbaw(mutations_selection)
            cmd.label("{} and name ca".format(mutations_selection), "'%s-%s' % (resn,resi)")
            cmd.zoom(mutations_selection)
    # record the image
    path_img = "{}_mutations.png".format(prefix)
    cmd.png(path_img, ray=1, quiet=1)
    logging.info("{} image: {}".format(cluster, path_img))

    # get the interface residues between each chain
    chains_interfaces = {}
    interfaces_ids = []
    chains = list(config.keys())
    for i in range(0, len(chains) - 1):
        prot_i = config[chains[i]]["name"]
        for j in range(i + 1, len(chains)):
            prot_j = config[chains[j]]["name"]
            interface_id = "interface_{}-{}".format(prot_i, prot_j)
            interface_raw = interfaceResidues.interfaceResidues(cluster, chains[i], chains[j], 1.0, interface_id)
            interface = {}
            for item in interface_raw:
                if item[0] in interface:
                    interface[item[0]][item[1]] = item[2]
                else:
                    interface[item[0]] = {item[1]: item[2]}
            interface[prot_i] = interface["chA"]
            del interface["chA"]
            interface[prot_j] = interface["chB"]
            del interface["chB"]
            chains_interfaces["{}_{}".format(prot_i, prot_j)] = interface

            # copy the interface to a new object
            cmd.create(interface_id, interface_id)
            interfaces_ids.append(interface_id)
            # disable the interface object for the pictures afterwards
            cmd.disable(interface_id)

    # save the interfaces images
    cmd.disable(cluster)
    for inter_id in interfaces_ids:
        cmd.enable(inter_id)
        path_img = "{}_{}.png".format(prefix, inter_id)
        cmd.png(path_img, ray=1, quiet=1)
        cmd.disable(inter_id)
        logging.info("{} image: {}".format(inter_id, path_img))
    print(chains_interfaces)

    return chains_interfaces


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
    interfaces = get_interfaces(cluster_id, args.input, args.prefix, configuration)

    #todo: heatmaps des interfaces
