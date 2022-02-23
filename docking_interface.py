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
from Bio.SeqUtils import seq1
from pymol import cmd
import yaml

sys.path.insert(0, "references")
import interfaceResidues, polarPairs


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
    Load the configuration file and update it with the path to pdb file and the chains sizes.

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
    path_pdb = os.path.join(pdb_dir, "{}.pdb".format(cluster))
    parser_pdb = PDB.PDBParser(QUIET=True)
    structure = parser_pdb.get_structure(cluster_id, path_pdb)
    model = structure[0]
    for chain in model:
        config["chains"]["chain {}".format(chain.id)]["length"] = len(list(chain.get_residues()))

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
    logging.info("[{}] {} detected with the minimal binding energy: "
                 "{}".format(get_cluster_min_binding_energy.__name__.replace("_", " ").title(),
                             cluster_min_energy, min_energy))
    return cluster_min_energy


def proteins_setup(cluster, input_dir, config, prefix):
    """
    Set colors to proteins and regions. Save an image.

    :param cluster: the cluster ID from HADDOCK outputs.
    :type cluster: str
    :param input_dir: the path to the HADDOCK outputs directory.
    :type input_dir: str
    :param config: the configuration of the chains for the analysis.
    :type config: dict
    :param prefix: the prefix of the outputs files.
    :type prefix: str
    :return: the image path.
    :rtype: str
    """

    cmd.do("load {}".format(os.path.join(input_dir, "{}.pdb".format(cluster))))
    cmd.show("cartoon", cluster)

    for chain in config:
        # set chain color
        cmd.color(config[chain]["color"], chain)
        logging.info("[{}] {} ({}): color set to {}.".format(proteins_setup.__name__.replace("_", " ").title(),
                                                             chain,
                                                             config[chain]["name"],
                                                             config[chain]["color"]))
        # set regions and colors if any
        if "regions" in config[chain]:
            for region_id, region_data in config[chain]["regions"].items():
                cmd.select(region_id, "{} and resi {}-{}".format(chain, region_data["start"], region_data["end"]))
                cmd.color(region_data["color"], region_id)
                logging.info("\tregion {}: from {} to {}, color set to {}.".format(region_id,
                                                                                   region_data["start"],
                                                                                   region_data["end"],
                                                                                   region_data["color"]))
    # record the image
    path_img = "{}.png".format(prefix)
    cmd.png(path_img, ray=1, quiet=1)

    return path_img


def highlight_mutations(config, prefix):
    """
    Update positions and highlight the mutations based on a reference sequence for each protein.

    :param config: the configuration of the chains for the analysis.
    :type config: dict
    :param prefix: the prefix of the outputs files.
    :type prefix: str
    """

    # highlight the mutations
    for chain in config:
        if "mutations" in config[chain]:
            logging.info("[{}] {} ({}):".format(highlight_mutations.__name__.replace("_", " ").title(),
                                                chain, config[chain]["name"]))
            # update index with alterations from the reference sequence where the mutations index come from
            if "alterations" in config[chain]["mutations"]:
                shift_idx = 0
                for alter_pos, alter_value in config[chain]["mutations"]["alterations"].items():
                    alter_str = "{}{}".format("+" if alter_value > 0 else "-", alter_value)
                    cmd.alter("{} and resi {}-{}".format(chain, alter_pos + shift_idx,
                                                         config[chain]["length"] + shift_idx),
                              "resi=str(int(resi){})".format(alter_str))
                    logging.info("\talteration of {} from position {} (original "
                                 "position {}).".format(alter_str, alter_pos + shift_idx, alter_pos))
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
            path_img = "{}_mutations-{}.png".format(prefix, config[chain]["name"])
            cmd.png(path_img, ray=1, quiet=1)
            logging.info("\t{} mutations image: {}".format(config[chain]["name"], path_img))

    path_updated_pdb = "{}_updated.pdb".format(prefix)
    cmd.save(path_updated_pdb, state=0)
    return path_updated_pdb


def get_residue_from_atom(atom_nb, atom, conf, distance=None):
    """
    Search data from the atom to retrieve the residue and position the atoms belongs to, the nature of the atoms and
    the distance from the first atom of the contact if the distance is provided.

    :param atom_nb: the serial number of the atom.
    :type atom_nb: int
    :param atom: the atom data.
    :type atom: str
    :param conf: the analysis configuration.
    :type conf: dict
    :param distance: the distance between the 2 atoms
    :type distance: float
    :return: the residue and the dictionary describing the residue with the chain, the atom nature and serial number
    and the distance if the atom is the second one in the interaction.
    :rtype: str, dict
    """
    chain_name = conf["chains"]["chain {}".format(atom.get_full_id()[2])]["name"]
    residue_pos = atom.get_full_id()[3][1]
    residue_id = "{}{}".format(seq1(atom.get_parent().get_resname()), residue_pos)
    logging.debug("\tatom{} belonging to residue: {}".format(1 if distance is None else 2, residue_id))
    if distance is None:
        data = {"description": {"chain": chain_name, "atom": {"type": atom.get_id(), "serial number": atom_nb}},
                "contacts with": {}}
    else:
        logging.debug("\tdistance from atom1: {:.2f} Angstroms".format(distance))
        data = {"chain": chain_name, "atom": {"type": atom.get_id(), "serial number": atom_nb}, "distance": distance}

    return residue_id, data


def get_contacts(model_id, config, chain1, chain2, contact_id):
    """
    Get the atoms in contact between 2 chains, retrieve the residues they belong to and the contact distance.
    The output dictionary has the following architecture:


    :param model_id: the model on which contacts are searched
    :type model_id:
    :param config: the configuration file of the analysis.
    :type config: dict
    :param chain1: the name of the first chain.
    :type chain1: str
    :param chain2: the name of the second chain.
    :type chain2: str
    :param contact_id: the name of the contact object.
    :type contact_id: str
    :return: the contact data.
    :rtype: dict
    """
    contacts = {}
    # get the atoms pairs contacts between chains
    raw_pairs_contacts = polarPairs.polar_pairs("{} and {}".format(model_id, chain1),
                                                "{} and {}".format(model_id, chain2),
                                                cutoff=config["contacts"]["cutoff"],
                                                angle=config["contacts"]["angle"],
                                                name=contact_id)
    # get the PDB structure
    parser_pdb = PDB.PDBParser(QUIET=True)
    structure = parser_pdb.get_structure(model_id, config["pdb"])
    model = structure[0]
    # search the atoms in contact
    for raw_pairs_contact in raw_pairs_contacts:
        atom1_serial_number = raw_pairs_contact[0]
        atom2_serial_number = raw_pairs_contact[1]
        dist = raw_pairs_contact[2]
        atom1 = None
        atom2 = None
        atoms_found = False
        logging.debug("raw pairs contact: {}".format(raw_pairs_contact))
        # get the data from the two contacts atoms
        for chain in model:
            for chain_atom in chain.get_atoms():
                if atom1_serial_number == chain_atom.get_serial_number():
                    atom1 = chain_atom
                    logging.debug("\tatom1 ({}) found in {} (chain {}) with full "
                                  "ID:\t{}".format(atom1_serial_number,
                                                   config["chains"]["chain {}".format(chain.id)]["name"],
                                                   chain.id, atom1.get_full_id()))
                if atom2_serial_number == chain_atom.get_serial_number():
                    atom2 = chain_atom
                    logging.debug("\tatom2 ({}) found in {} (chain {}) with full"
                                  "ID:\t{}".format(atom2_serial_number,
                                                   config["chains"]["chain {}".format(chain.id)]["name"],
                                                   chain.id, atom2.get_full_id()))
                if atom1 is not None and atom2 is not None:
                    atoms_found = True
                    break
            if atoms_found:
                break

        # set data for atom1
        residue1, residue_info = get_residue_from_atom(atom1_serial_number, atom1, config)
        if residue1 not in contacts:
            contacts[residue1] = residue_info
        # set data for atom2
        residue2, residue2_info = get_residue_from_atom(atom2_serial_number, atom2, config, distance=dist)
        contacts[residue1]["contacts with"][residue2] = residue2_info

    return contacts


def get_interactions(cluster, prefix, config):
    """
    Get the interactions of two types between the proteins:
        - the residues at the interface for each chain by couple of proteins
        - the contacts residues between the chains for each couple of proteins.

    :param cluster: the cluster ID from HADDOCK outputs.
    :type cluster: str
    :param prefix: the prefix of the outputs files.
    :type prefix: str
    :param config: the whole configuration of the analysis.
    :type config: dict
    :return: the interactions.
    :rtype: dict
    """

    # get the interface residues between each chain
    interactions = {}
    chains = list(config["chains"].keys())
    for i in range(0, len(chains) - 1):
        prot_i = config["chains"][chains[i]]["name"]
        for j in range(i + 1, len(chains)):
            prot_j = config["chains"][chains[j]]["name"]
            protein_couple = "{}-{}".format(prot_i, prot_j)
            logging.info("[{}] {}".format(get_interactions.__name__.replace("_", " ").title(), protein_couple))
            interface_id = "interface_{}".format(protein_couple)
            interface_selection = "{}_sele".format(interface_id)
            interface_raw = interfaceResidues.interfaceResidues(cluster, chains[i], chains[j], 1.0, interface_selection)
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
            interactions[protein_couple] = {"interface": interface}

            # copy the interface to a new object
            cmd.create(interface_id, interface_selection)
            cmd.zoom(interface_id)
            # disable cluster view to have only interface view
            cmd.disable(cluster)
            path_img = "{}_{}.png".format(prefix, interface_id)
            cmd.png(path_img, ray=1, quiet=1)
            logging.info("\t{} image: {}".format(interface_id.replace("_", " "), path_img))
            cmd.enable(cluster)

            interactions[protein_couple]["contacts"] = get_contacts(cluster, config, chains[i], chains[j],
                                                                    "contacts_{}-{}".format(prot_i, prot_j))
            cmd.disable(interface_id)

    return interactions


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
    parser.add_argument("-b", "--background-images", required=False, action="store_true",
                        help="set an opaque background for the images.")
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

    # set background for images
    if args.background_images:
        cmd.set("ray_opaque_background", 1)
        logging.info("background set to opaque for images.")

    # get the pdb cluster file with the minimal binding energy
    cluster_id = get_cluster_min_binding_energy(args.input)

    # load the configuration file and update it
    configuration = update_config(args.config, cluster_id, args.input)

    # set up the proteins and regions
    img_path = proteins_setup(cluster_id, args.input, configuration["chains"], args.prefix)
    logging.info("{} image: {}".format(cluster_id, img_path))

    # highlight mutations
    updated_pdb_path = highlight_mutations(configuration["chains"], args.prefix)
    configuration["pdb"] = updated_pdb_path

    # load the pdb file from the updated configuration file, get the interfaces and the contacts between residues
    interactions_between_chains = get_interactions(cluster_id, args.prefix, configuration)
    for k in interactions_between_chains:
        for r1 in interactions_between_chains[k]["contacts"]:
            print("{}: {}".format(r1, interactions_between_chains[k]["contacts"][r1]["description"]))
            for r2 in interactions_between_chains[k]["contacts"][r1]["contacts with"]:
                print("\t{}: {}".format(r2, interactions_between_chains[k]["contacts"][r1]["contacts with"][r2]))

    # save the session
    cmd.save("{}.pse".format(args.prefix), state=0)

