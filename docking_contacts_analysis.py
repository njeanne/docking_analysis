#!/usr/bin/env python3

"""
Created on 17 Feb. 2022
"""

__author__ = "Nicolas JEANNE"
__copyright__ = "GNU General Public License"
__email__ = "jeanne.n@chu-toulouse.fr"
__version__ = "1.1.1"

import argparse
import logging
import os
import re
import sys

import altair as alt
import numpy as np
import pandas as pd
import yaml
from Bio import PDB
from Bio.SeqUtils import seq1
from pymol import cmd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "references"))
import interfaceResidues
import polarPairs


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
    logging.info(f"[{get_cluster_min_binding_energy.__name__.replace('_', ' ').title()}] {cluster_min_energy} detected "
                 f"with the minimal binding energy: {min_energy}")
    return cluster_min_energy


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
    config["pdb"] = os.path.join(pdb_dir, f"{cluster}.pdb")
    parser_pdb = PDB.PDBParser(QUIET=True)
    structure = parser_pdb.get_structure(cluster_id, config["pdb"])
    model = structure[0]
    for chain in model:
        config["chains"][f"chain {chain.id}"]["length"] = len(list(chain.get_residues()))

    return config


def proteins_setup(cluster, config, out_dir, involved_prot):
    """
    Set colors to proteins and regions. Save an image.

    :param cluster: the cluster ID from HADDOCK outputs.
    :type cluster: str
    :param config: the configuration for the analysis.
    :type config: dict
    :param out_dir: the output directory.
    :type out_dir: str
    :param involved_prot: the proteins names involved in the whole docking.
    :type involved_prot: str
    :return: the image path.
    :rtype: str
    """

    cmd.do(f"load {config['pdb']}")
    cmd.show("cartoon", cluster)

    for chain in config["chains"]:
        # set chain color
        cmd.color(config["chains"][chain]["color"], chain)
        logging.info(f"[{proteins_setup.__name__.replace('_', ' ').title()}] {chain} "
                     f"({config['chains'][chain]['name']}): color set to {config['chains'][chain]['color']}.")
        # set regions and colors if any
        if "regions" in config["chains"][chain]:
            for region_id, region_data in config["chains"][chain]["regions"].items():
                cmd.select(region_id, f"{chain} and resi {region_data['start']}-{region_data['end']}")
                cmd.color(region_data["color"], region_id)
                logging.info(f"\tregion {region_id}: from {region_data['start']} to {region_data['end']}, color set to "
                             f"{region_data['color']}.")
    # record the image
    path_img = os.path.join(out_dir, f"{involved_prot}.png")
    cmd.png(path_img, ray=1, quiet=1)

    return path_img


def licorice_residues(positions, pdb_chain_id, sele):
    """
    Set residues as licorice.

    :param positions: the list of positions to highlight
    :type positions: list
    :param pdb_chain_id: the identification of the chain as in the PDB file.
    :type pdb_chain_id: str
    :param sele: the name of the selection.
    :type sele: str
    """

    # prepare the residue selection
    sele_str = positions[0]
    for idx in range(1, len(positions)):
        sele_str = f"{sele_str}+{positions[idx]}"
    cmd.select(sele, f"{pdb_chain_id} and resi {sele_str}")
    cmd.show("licorice", sele)
    cmd.label(f"{sele} and name ca", "'%s-%s' % (resn,resi)")
    cmd.disable(sele)


def highlight_positions_of_interest(config, out_dir, involved_prot):
    """
    Update positions and highlight the positions of interest based on a reference sequence for each protein.

    :param config: the configuration for the analysis.
    :type config: dict
    :param out_dir: the output directory.
    :type out_dir: str
    :param involved_prot: the proteins names involved in the whole docking.
    :type involved_prot: str
    :return: the path of the pdb where the positions were updated.
    :rtype: str
    """

    # highlight the POI
    for chain in config["chains"]:
        if "POI" in config["chains"][chain]:
            logging.info(f"[{highlight_positions_of_interest.__name__.replace('_', ' ').title()}] {chain} "
                         f"({config['chains'][chain]['name']}):")
            # update index with alterations from the reference sequence where the positions of interest index come from
            if "alterations" in config["chains"][chain]["POI"]:
                shift_idx = 0
                for alter_pos, alter_value in config["chains"][chain]["POI"]["alterations"].items():
                    shifted_pos = alter_pos + shift_idx
                    cmd.alter(f"{chain} and resi {shifted_pos}-{config['chains'][chain]['length'] + shift_idx}",
                              f"resi=str(int(resi){'+' if alter_value > 0 else ''}{alter_value})")
                    logging.info(f"\talteration of {alter_value} from position {shifted_pos} (original "
                                 f"position {alter_pos}).")
                    shift_idx = shift_idx + alter_value

            # highlight the POIs
            licorice_residues(config["chains"][chain]["POI"]["positions"], chain,
                              f"{config['chains'][chain]['name'].replace(' ', '_')}_POI")
            logging.info(f"\t{config['chains'][chain]['name']} residues of interest highlighted.")

    path_highlighted_poi_pdb = os.path.join(out_dir, f"{involved_prot}_updated.pdb")
    cmd.save(path_highlighted_poi_pdb, state=0)

    return path_highlighted_poi_pdb


def get_interface_view(chain1, chain2, id_cluster, id_couple, out_dir, prot1, prot2):
    """
    Get the interface and save it as an image.

    :param chain1: the chain 1 ID.
    :type chain1: str
    :param chain2: the chain 2 ID.
    :type chain2: str
    :param id_cluster: the cluster ID.
    :type id_cluster: str
    :param id_couple: the protein couple name.
    :type id_couple: str
    :param out_dir: the output directory path.
    :type out_dir: str
    :param prot1: the protein 1 name.
    :type prot1: str
    :param prot2: the protein 2 name.
    :type prot2: str
    :return: the interface residues.
    :rtype: dict
    """
    interface_id = f"interface_{id_couple}"
    interface_selection = f"{interface_id}_sele"
    interface_raw = interfaceResidues.interfaceResidues(cluster, chain1, chain2, 1.0, interface_selection)
    interface = {}
    for item in interface_raw:
        if item[0] in interface:
            interface[item[0]][item[1]] = item[2]
        else:
            interface[item[0]] = {item[1]: item[2]}
    interface[prot1] = interface["chA"]
    del interface["chA"]
    interface[prot2] = interface["chB"]
    del interface["chB"]
    # copy the interface to a new object
    cmd.create(interface_id, interface_selection)
    cmd.zoom(interface_id)
    # disable cluster view to have only interface view for the image
    cmd.disable(id_cluster)
    path_img = os.path.join(out_dir, f"{interface_id}.png")
    cmd.png(path_img, ray=1, quiet=1)
    cmd.disable(interface_id)
    cmd.enable(id_cluster)
    logging.info(f"\tinterface {id_couple} image: {path_img}")

    return interface


def get_residue_from_atom(atom, conf, first):
    """
    Search data from the atom to retrieve the residue and position the atom belongs to, the nature and the contact
    distance for the atom of the other residue if the distance is provided.

    :param atom: the atom data.
    :type atom: Bio.PDB.Atom
    :param conf: the analysis configuration.
    :type conf: dict
    :param first: if it is the first of the 2 contact atoms
    :type first: bool
    :return: the tuple of the residue position and the residue type, the dictionary describing the residue with the
    chain, the atom nature and serial number and the distance if the atom is the second one in the interaction.
    :rtype: tuple, dict
    """
    id_chain = atom.get_full_id()[2]
    chain_name = conf["chains"][f"chain {id_chain}"]["name"]
    residue_position = atom.get_full_id()[3][1]
    residue_type = seq1(atom.get_parent().get_resname())
    logging.debug(f"\tatom{1 if first else 2} {atom.get_id()} belonging to residue: {residue_type}")
    data = {"chain": chain_name}
    if first:
        data["contacts with"] = {}

    return (residue_position, residue_type), data


def get_atom_serial_number(model, searched_chain, searched_atom_serial_nb, conf, atom_nb_id):
    """
    Search the contact atom in the correct chain.

    :param model: the model containing the list of chains.
    :type model: list
    :param searched_chain: the identifier of the searched chain.
    :type searched_chain: str
    :param searched_atom_serial_nb: the serial number of the contact atom in the chain.
    :type searched_atom_serial_nb: int
    :param conf: the configuration of the analysis.
    :type conf: dict
    :param atom_nb_id: the atom identifier for method debugging.
    :type atom_nb_id: str
    :return: the atom in the chain.
    :rtype: Bio.PDB.Atom
    """
    atom = None
    for chain in model:
        if f"chain {chain.id}" == searched_chain:
            for chain_atom in chain.get_atoms():
                if searched_atom_serial_nb == chain_atom.get_serial_number():
                    atom = chain_atom
                    found_chain_name = conf["chains"][f"chain {chain.id}"]["name"]
                    logging.debug(f"\t{atom_nb_id} ({searched_atom_serial_nb}) found in {found_chain_name} "
                                  f"(chain {chain.id}) with full ID:\t{atom.get_full_id()}")
                    return atom
    return atom


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
    raw_pairs_contacts = polarPairs.polar_pairs(f"{model_id} and {chain1}",
                                                f"{model_id} and {chain2}",
                                                cutoff=config["contacts"]["cutoff"],
                                                angle=config["contacts"]["angle"],
                                                name=contact_id)
    # get the PDB structure
    parser_pdb = PDB.PDBParser(QUIET=True)
    structure = parser_pdb.get_structure(model_id, config["updated pdb"])
    model_pdb = structure[0]
    # search the atoms in contact
    for raw_pairs_contact in raw_pairs_contacts:
        dist = raw_pairs_contact[2]
        logging.debug(f"raw pairs contact: {raw_pairs_contact}")

        # get the data from the two contacts atoms
        atom1 = get_atom_serial_number(model_pdb, chain1, raw_pairs_contact[0], config, "atom1")
        atom2 = get_atom_serial_number(model_pdb, chain2, raw_pairs_contact[1], config, "atom2")

        # set data for atom1
        res1_tuple, res1_data = get_residue_from_atom(atom1, config, first=True)
        if res1_tuple not in contacts:
            contacts[res1_tuple] = res1_data
        # set data for atom2
        res2_tuple, res2_data = get_residue_from_atom(atom2, config, first=False)
        if res2_tuple in contacts[res1_tuple]["contacts with"]:
            contacts[res1_tuple]["contacts with"][res2_tuple]["distances"].append(dist)
        else:
            res2_data["distances"] = [dist]
            contacts[res1_tuple]["contacts with"][res2_tuple] = res2_data
        logging.debug(f"\tdistance from atom1: {dist:.2f} Angstroms")

    return contacts


def get_interactions(cluster, out_dir, config, get_interfaces):
    """
    Get the interactions of two types between the proteins:
        - the residues at the interface for each chain by couple of proteins
        - the contacts residues between the chains for each couple of proteins.

    :param cluster: the cluster ID from HADDOCK outputs.
    :type cluster: str
    :param out_dir: the output directory.
    :type out_dir: str
    :param config: the whole configuration of the analysis.
    :type config: dict
    :param get_interfaces: if the interfaces residues should be computed.
    :type get_interfaces: bool
    :return: the interactions.
    :rtype: dict
    """

    # get the interface residues between each chain
    interactions = {}
    contacts_to_highlight = {}
    chains = list(config["chains"].keys())
    for i in range(0, len(chains) - 1):
        if chains[i] not in contacts_to_highlight:
            contacts_to_highlight[chains[i]] = set()
        prot_i = config["chains"][chains[i]]["name"].replace(" ", "-")
        for j in range(i + 1, len(chains)):
            if chains[j] not in contacts_to_highlight:
                contacts_to_highlight[chains[j]] = set()
            prot_j = config["chains"][chains[j]]["name"].replace(" ", "-")
            couple = f"{prot_i}-{prot_j}"
            interactions[couple] = {}
            logging.info(f"[{get_interactions.__name__.replace('_', ' ').title()}] {couple}")
            # get the interface if required
            if get_interfaces:
                interactions[couple] = get_interface_view(chains[i], chains[j], cluster, couple, out_dir, prot_i,
                                                          prot_j)

            # get the contacts
            interactions[couple]["contacts"] = get_contacts(cluster, config, chains[i], chains[j], f"contacts_{couple}")

            # record the contacts to highlight them
            for tuple_i in interactions[couple]["contacts"]:
                contacts_to_highlight[chains[i]].add(tuple_i[0])
                for tuple_j in interactions[couple]["contacts"][tuple_i]["contacts with"]:
                    contacts_to_highlight[chains[j]].add(tuple_j[0])

    # licorice the contacts and colorize the contacts if intersection with POI
    for chain in chains:
        licorice_residues(sorted(list(contacts_to_highlight[chain])), chain,
                          f"{config['chains'][chain]['name'].replace(' ', '_')}_contacts")
        if "POI" in config["chains"][chain] and "positions" in config["chains"][chain]["POI"]:
            intersection = list(contacts_to_highlight[chain] & set(config["chains"][chain]["POI"]["positions"]))
            logging.info(f"\t{config['chains'][chain]['name']} has {len(intersection)} contacts residues in Positions "
                         f"of Interest.")
        else:
            intersection = None
            logging.info(f"\t{config['chains'][chain]['name']} has no Positions of Interest set in the configuration "
                         f"file.")
        if intersection:
            sele_color = f"{chain} and resi {intersection[0]}"
            for idx in range(1, len(intersection)):
                sele_color = f"{sele_color},{intersection[idx]}"
            cmd.color(config["chains"][chain]["POI"]["contact color"], sele_color)

    return interactions


def contacts_heatmap(data, couple, config, out_dir):
    """
    Create the contacts heatmap plot.

    :param data: the contacts data.
    :type data: dict
    :param couple: the proteins involved.
    :type couple: str
    :param config: the whole configuration of the analysis.
    :type config: dict
    :param out_dir: the output directory.
    :type out_dir: str
    """

    # create the input dataframe
    contacts1 = list()
    contacts2 = set()
    residues_contacts = {}
    # set up variables to get the last tuples to get access after the for loop to the chains IDs
    tuple1 = None
    tuple2 = None
    for tuple1 in data:
        t1_str = f"{tuple1[0]}{tuple1[1]}"
        contacts1.append(tuple1)
        residues_contacts[t1_str] = {}
        for tuple2 in data[tuple1]["contacts with"]:
            t2_str = f"{tuple2[0]}{tuple2[1]}"
            contacts2.add(tuple2)
            residues_contacts[t1_str][t2_str] = data[tuple1]["contacts with"][tuple2]["distances"]
    chain1 = data[tuple1]["chain"]
    chain2 = data[tuple1]["contacts with"][tuple2]["chain"]
    contacts1 = sorted(contacts1)
    contacts2 = sorted(list(contacts2))

    # create the meshgrid to prepare the dataframe
    x, y = np.meshgrid([f"{t[0]}{t[1]}" for t in contacts1], [f"{t[0]}{t[1]}" for t in contacts2])
    # create the contact distance list with the minimal distances in the list of distances between 2 residues.
    min_distances = list()
    nb_contacts = list()
    for tuple2 in contacts2:
        for tuple1 in contacts1:
            if tuple2 in data[tuple1]["contacts with"]:
                min_distances.append(min(data[tuple1]["contacts with"][tuple2]["distances"]))
                nb_contacts.append(len(data[tuple1]["contacts with"][tuple2]["distances"]))
            else:
                min_distances.append(np.nan)
                nb_contacts.append(np.nan)

    # Convert this grid to columnar data expected by Altair
    source = pd.DataFrame({chain1.replace(".", "_"): x.ravel(), chain2.replace(".", "_"): y.ravel(),
                           "minimal_contact_distance": min_distances, "number_of_contacts": nb_contacts})
    out_path_df = os.path.join(out_dir, f"{couple}_contacts.csv")
    source.to_csv(out_path_df, index=True)

    # create the heatmap
    heatmap = alt.Chart(data=source).mark_rect().encode(
        x=alt.X(chain1.replace(".", "_"), title=chain1),
        y=alt.Y(chain2.replace(".", "_"), title=chain2, sort=None),
        color=alt.Color("minimal_contact_distance:Q", title="Distance (\u212B)", sort="descending",
                        scale=alt.Scale(scheme="yelloworangered"))
    ).properties(
        title={
            "text": f"{couple}: contact residues",
            "subtitle": [f"Threshold {config['contacts']['cutoff']} Angstroms",
                         "Number of contacts displayed in the squares"],
            "subtitleColor": "gray"
        },
        width=config["heatmap"]["width"],
        height=config["heatmap"]["height"]
    )
    # Configure the text with the number of contacts
    distances_values = [v for v in source["minimal_contact_distance"] if not np.isnan(v)]
    switch_color = min(distances_values) + (max(distances_values) - min(distances_values)) / 2
    text = heatmap.mark_text(baseline="middle").encode(
        text=alt.Text("number_of_contacts"),
        color=alt.condition(
            alt.datum.minimal_contact_distance > switch_color,
            alt.value("black"),
            alt.value("white")
        )
    )
    plot = heatmap + text

    # save the plot
    out_path_plot = os.path.join(out_dir, f"{couple}_contacts.{config['heatmap']['format']}")
    plot.save(out_path_plot)
    logging.info(f"[{contacts_heatmap.__name__.replace('_', ' ').title()}] {couple} contacts heatmap: {out_path_plot}")


if __name__ == "__main__":
    descr = f"""
    {os.path.basename(__file__)} v. {__version__}

    Created by {__author__}.
    Contact: {__email__}
    {__copyright__}

    Distributed on an "AS IS" basis without warranties or conditions of any kind, either express or implied.

    From a docking performed by HADDOCK, select the better cluster and extract the interface atoms.
    """
    parser = argparse.ArgumentParser(description=descr, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", required=True, type=str, help="the path to the output directory.")
    parser.add_argument("-c", "--config", required=True, type=str, help="the path to the YAML configuration file.")
    parser.add_argument("-i", "--interfaces", required=False, action="store_true",
                        help="get the interfaces between the proteins.")
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
                        help="the path to the HADDOCK pdb output directory or directly to the file to use.")
    args = parser.parse_args()

    # create output directory if necessary
    os.makedirs(args.out, exist_ok=True)

    # create the logger
    if args.log:
        log_path = args.log
    else:
        log_path = os.path.join(args.out, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
    create_log(log_path, args.log_level)

    logging.info(f"version: {__version__}")
    logging.info(f"CMD: {' '.join(sys.argv)}")

    # set background for images
    if args.background_images:
        cmd.set("ray_opaque_background", 1)
        logging.info("background set to opaque for images.")

    # lok if the input is a file or a directory and update the config file
    if os.path.isfile(args.input):
        # no search for cluster, set the file name as cluster
        cluster_id = os.path.splitext(os.path.basename(args.input))[0]
        configuration = update_config(args.config, cluster_id, os.path.dirname(args.input))
    else:
        # get the pdb cluster file with the minimal binding energy
        cluster_id = get_cluster_min_binding_energy(args.input)
        configuration = update_config(args.config, cluster_id, args.input)

    # set up the proteins and regions
    proteins_involved = []
    for chain_id in configuration["chains"]:
        proteins_involved.append(configuration["chains"][chain_id]["name"])
    proteins_involved = "-".join(proteins_involved)
    img_path = proteins_setup(cluster_id, configuration, args.out, proteins_involved)
    logging.info(f"{cluster_id} image: {img_path}")

    # highlight mutations
    updated_pdb_path = highlight_positions_of_interest(configuration, args.out, proteins_involved)
    configuration["updated pdb"] = updated_pdb_path

    # load the pdb file from the updated configuration file, get the interfaces and the contacts between residues
    interactions_between_chains = get_interactions(cluster_id, args.out, configuration, args.interfaces)

    # create the contacts heatmap plot
    for protein_couple in interactions_between_chains:
        contacts_heatmap(interactions_between_chains[protein_couple]["contacts"],
                         protein_couple,
                         configuration,
                         args.out)

    # save the session
    cmd.save(os.path.join(args.out, f"{proteins_involved}_contacts.pse"), state=0)
