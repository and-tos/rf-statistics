#!/usr/bin/env python

########################################################################################################################

# Use ccdc_babel to combine two databases.
# ccdc_babel -auto <path_to_new_database.csdsql> -csdsql3 <path_to_existing_database.csdsql>

from ccdc import io, molecule, protein, entry
from ccdc.protein import Protein
from pathlib import Path
import argparse
import pandas as pd
from io import StringIO

########################################################################################################################


def parse_args():
    """Define and parse the arguments to the script."""
    parser = argparse.ArgumentParser(
        description="""
        Generate CSDSQL database from proasis mol2 files.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # To display default values in help message.
    )

    parser.add_argument(
        "-i",
        "--input",
        default="public",
        choices=["public", "internal"],
    )

    return parser.parse_args()


def mol2_to_df(mol2_string):
    record_types = mol2_string.split("@<TRIPOS>")
    atoms_string = "\n".join(record_types[2].split("\n")[1:])
    bonds_string = "\n".join(record_types[3].split("\n")[1:])
    mol_string = "\n".join(record_types[1].split("\n")[1:])
    for prefix in ["_ZN", "_UN", "_ZO", "_UO", "_ZC", "_UC", "_ZCl", "_UCl"]:
        atoms_string = atoms_string.replace(prefix + " ", prefix)
    atoms_df = pd.read_csv(StringIO(atoms_string), delim_whitespace=True, header=None)

    return atoms_df, bonds_string, mol_string


def build_new_mol2(relabel_atoms_df, mol_string, bonds_string):
    substructure_str = "@<TRIPOS>SUBSTRUCTURE\n"
    group_columns = ["residue_label", "chain_id"]
    for resid, group in enumerate(relabel_atoms_df.groupby(group_columns, sort=False)):
        residue_chain, group_df = group
        residue, chain_id = residue_chain
        base_atom = group_df["atom_id"].to_numpy()[0]
        split_res = residue.split("_")
        resname = split_res[0][:3]
        resnum = split_res[0][3:]

        if chain_id == "":
            chain_id = "XX"
        substructure_str = (
            substructure_str
            + f"{resid + 1} {resname}{resnum} {base_atom} RESIDUE 1 {chain_id} {resname}\n"
        )

    mol_string = mol_string.split("\n")
    mol_prop_line = mol_string[1].split()
    mol_prop_line[2] = str(resid + 1)
    mol_prop_line = " ".join(mol_prop_line)
    mol_string[1] = mol_prop_line
    mol_string = "\n".join(mol_string)

    atoms_string = relabel_atoms_df.drop(columns="chain_id").to_string(
        header=False, index=False
    )

    mol2_string = "@<TRIPOS>MOLECULE\n" + mol_string
    mol2_string = mol2_string + "@<TRIPOS>ATOM\n" + atoms_string + "\n"
    mol2_string = mol2_string + "@<TRIPOS>BOND\n" + bonds_string
    mol2_string = mol2_string + substructure_str

    return mol2_string


def relabel_moe_with_proasis(proasis_mol2, relabel_mol2):
    proasis_atoms_df = mol2_to_df(proasis_mol2.to_string("mol2"))[0]
    proasis_atoms_df = proasis_atoms_df.drop(columns=[8])
    proasis_atoms_df.columns = [
        "atom_id",
        "label",
        "x",
        "y",
        "z",
        "sybyl_type",
        "resnum",
        "residue_label",
    ]
    for a in proasis_mol2.atoms:
        proasis_atoms_df.loc[proasis_atoms_df["label"] == a.label, "chain_id"] = (
            a.chain_label
        )
    try:
        split_res_df = proasis_atoms_df["residue_label"].str.split("_", expand=True)
        if split_res_df.shape[1] == 2:
            proasis_atoms_df[["residue_label", "chain_id"]] = split_res_df
    except ValueError as e:
        print(e)

    relabel_atoms_df, bonds_string, mol_string = mol2_to_df(
        relabel_mol2.to_string("mol2")
    )
    relabel_atoms_df = relabel_atoms_df.drop(columns=[6, 7])
    relabel_atoms_df.columns = [
        "atom_id",
        "label",
        "x",
        "y",
        "z",
        "sybyl_type",
        "charge",
    ]
    relabel_atoms_df = relabel_atoms_df.join(
        proasis_atoms_df[["label", "resnum", "residue_label", "chain_id"]].set_index(
            "label"
        ),
        on="label",
    )
    relabel_atoms_df = relabel_atoms_df[
        [
            "atom_id",
            "label",
            "x",
            "y",
            "z",
            "sybyl_type",
            "resnum",
            "residue_label",
            "chain_id",
            "charge",
        ]
    ]
    for index, row in relabel_atoms_df.iterrows():
        if row["label"].startswith("H") and not row["label"].startswith("HO"):
            relabel_atoms_df.loc[index, ["resnum", "residue_label", "chain_id"]] = (
                relabel_atoms_df.loc[index - 1, ["resnum", "residue_label", "chain_id"]]
            )
        else:
            continue
    relabel_atoms_df = relabel_atoms_df.astype({"resnum": int})
    mol2_string = build_new_mol2(relabel_atoms_df, mol_string, bonds_string)

    ccdc_mol = protein.Protein.from_entry(entry.Entry.from_string(mol2_string))

    # normalise water labels
    ccdc_mol_norm = ccdc_mol.copy()
    # ccdc_mol_norm.normalise_labels()
    for i, a in enumerate(ccdc_mol_norm.atoms):
        if not a.protein_atom_type == "Water":
            a.label = ccdc_mol.atoms[i].label

    return ccdc_mol


def fix_mol(mol2_file):
    m3d_components = io.MoleculeReader(mol2_file)
    bs_id = m3d_components[0].identifier
    strucid = bs_id.split("_")[0].lower()
    proasis_mol2 = f"{strucid}.mol2"
    proasis_mol = io.MoleculeReader(proasis_mol2).molecule(bs_id)
    proasis_mol_labels = [a.label for a in proasis_mol.atoms]
    proasis_mol.normalise_labels()
    for cnt, a in enumerate(proasis_mol.heavy_atoms):
        a.label = proasis_mol_labels[cnt][:2] + a.label

    m3d = molecule.Molecule()
    for c in m3d_components:
        m3d.add_molecule(c)

    m3d.identifier = c.identifier.split(".")[0]

    for cnt, a in enumerate(m3d.heavy_atoms):
        a.label = proasis_mol.heavy_atoms[cnt].label

    relabel_mol2 = str(Path(mol2_file).parent / f"{bs_id}_protonate3d_relabel_fix.mol2")
    relabel_csdsql = str(
        Path(mol2_file).parent / f"{bs_id}_protonate3d_relabel_fix.csdsql"
    )

    # set substructure records in mol2 file
    mol = relabel_moe_with_proasis(proasis_mol, m3d)

    # write mol2 in compatible format
    with io.MoleculeWriter(relabel_mol2) as w:
        w.write(mol)
    with io.MoleculeWriter(relabel_csdsql) as w:
        w.write(mol)
    return


def proasis_to_csdsql(
    path_to_proasis_mol2,
    output="database_.csdsql",
    old_csdsql=None,
    strucid_len=4,
    overwrite=False,
):
    if old_csdsql:
        print("getting old identifiers...")
        old_identifiers = [e.identifier for e in io.EntryReader(old_csdsql)]

    print("getting mol2 files")
    bs_path = Path(path_to_proasis_mol2)

    filestr = (
        "".join(["?" for i in range(strucid_len)]) + "_out/*_protonate3d_relabel.mol2"
    )

    mol2_files = bs_path.glob(filestr)

    identifiers = []
    if Path(output).is_file():
        identifiers = [mol.identifier for mol in io.MoleculeReader(output)]

    if overwrite:
        print("Writing new csdsql...")
        append = False
    else:
        print("Appending to existing csdsql file.")
        append = True
    with io.MoleculeWriter(output, append=append) as w:
        for mol2_file in mol2_files:
            ccdc_mol = Protein.from_file(str(mol2_file))
            molid = ccdc_mol.identifier
            if molid in identifiers:
                continue
            if old_csdsql and molid in old_identifiers:
                continue
            identifiers.append(ccdc_mol.identifier)
            w.write(ccdc_mol)
    return


def main():
    args = parse_args()
    old_csdsql = None
    datestring = "2024-12-03"
    if args.input == "public":
        strucid_len = 4
        output = f"full_p2cq_pub_{datestring}.csdsql"
        # old_csdsql = f"full_pub_concat.csdsql"

    if args.input == "internal":
        strucid_len = 5
        output = f"full_p2cq_roche_{datestring}.csdsql"
        # old_csdsql = f"full_p2cq_roche_2024-12-03.csdsql"
    proasis_to_csdsql("PPMOL2_SYM", output, old_csdsql, strucid_len=strucid_len)


if __name__ == "__main__":
    main()
