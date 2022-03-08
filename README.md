# Docking Contacts Analysis

From a docking experiment, look for the contacts between all the chains of the PDB input file(s) and if asked the interfaces between the chains.

## Conda environment

A `conda` YAML environment file is provided: `references/pymol_env.yml`. The file contains all the dependencies to run the script.
The conda environment is generated using the command:
```shell script
# create the environment
conda env create -f references/pymol_env.yml

# activate the environment
conda activate pymol
```

## Configuration file

A configuration file must be provided as the [template](references/config_template.yml) in the repository.

## Results

The outputs of the script are by pair of chains:
- a `png` file of the docking with the contacts.
- a `png` file of the docking interfaces if asked.
- a `pymol` session of the docking with the contacts (and the interfaces if asked).
- an updated `pdb` file.
- a heatmap of the contacts which format is set in the configuration file.
- a `csv` file of the contacts.
- a `log` file.