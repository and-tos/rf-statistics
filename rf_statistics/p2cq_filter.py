#! /usr/bin/env python

# program filters complex.csv and los.csv by resolution, real-space correlation coefficient and redundancy
# where redundancy is defined as: same project & same smiles. Best resolution is kept
# A. Tosstorff, B. Kuhn
# 30-OCT-2014

########################################################################################################################

import __future__
from pathlib import Path
import pandas as pd
import argparse
from ccdc import protein, io, descriptors, search
from rf_statistics import utils, atom_types, structure_quality
from rdkit import Chem
from rdkit.Chem import PandasTools
from itertools import chain
import tempfile
import time

########################################################################################################################


def parse_args():
    """Define and parse the arguments to the script."""
    parser = argparse.ArgumentParser(
        description="""
        Filter complex.csv and los.csv.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # To display default values in help message.
    )

    parser.add_argument(
        "-i", "-in", "--input", help="Path to input folder.", default="."
    )

    parser.add_argument(
        "-o", "-out", "--output", help="Output filename.", default="los_filtered.csv"
    )

    parser.add_argument("--rf_home", help="Path to database folder.", default=".")

    parser.add_argument("-r", "--resolution", help="Resolution cut-off.", default="2.5")

    parser.add_argument("-rscc", "--rscc", "--RSCC", help="RSCC cut-off.", default=0.8)

    parser.add_argument(
        "-m", "--mode", help="ligand or protein perspective.", default="ligand"
    )

    parser.add_argument(
        "-occu", "--occu", "--occupancy", help="Ligand avgoccu cut-off.", default=1
    )

    parser.add_argument(
        "-sq",
        "--structure_quality",
        help="File containing structure quality information.",
        default=None,
    )

    parser.add_argument(
        "--not_filter_by_cofactor",
        help="If passed, any entries where the ligand is a cofactor are not filtered out",
        action="store_true",
    )

    return parser.parse_args()


def protein_atom_makes_mcs_contact(entry_los_df, mcs_atoms):
    mcs_atom_indices = [int(i[1].label.split("_Z")[1]) + 1 for i in mcs_atoms]
    if entry_los_df[entry_los_df["los_atom_id"].isin(mcs_atom_indices)].shape[0] != 0:
        return True
    else:
        return False


def _mcs_includes_central_atom(
    query_ligand, reference_ligand, mode, query_central_atom_labels, entry_los_df
):
    """
    For ligand perspective filtering, an entry is considered redundant, if the ligand query atom is part of the MCS.
    For protein perspective filtering, an entry is considered redundant, if the protein query atom makes a contact to
    the MCS.
    :param query_ligand:
    :param reference_ligand:
    :param mode:
    :param query_central_atom_labels:
    :param entry_los_df:
    :return:
    """
    searcher = descriptors.MolecularDescriptors.MaximumCommonSubstructure()
    searcher.settings.ignore_hydrogens = True
    searcher.settings.check_hydrogen_count = True

    print("starting mcs search...")
    print("query: ", query_ligand.identifier)
    print("ref: ", reference_ligand.identifier)

    # this is for edge cases that will slow things down
    if len(reference_ligand.heavy_atoms) > 75:
        sim_searcher = search.SimilaritySearch(reference_ligand)
        sim = sim_searcher.search_molecule(query_ligand)
        if sim.similarity < 0.7:
            return False

    t1 = time.time()
    mcs = searcher.search(reference_ligand, query_ligand, search_step_limit=1000000)
    print("mcs search time: ", time.time() - t1)
    mcs_atoms = mcs[0]
    sub_bonds = [a[0] for a in mcs[1]]
    sub_atoms = [a[0] for a in mcs[0]]
    if len(mcs_atoms) < 15:
        return False
    if mode == "protein":
        if protein_atom_makes_mcs_contact(entry_los_df, mcs_atoms):
            return sub_bonds, sub_atoms
        else:
            return False
    else:
        for mcs_atom_pair in mcs_atoms:
            if mcs_atom_pair[1].label in query_central_atom_labels:
                return sub_bonds, sub_atoms
        return False


# def entry_mcs_is_redundant(
#     identifier,
#     project,
#     filtered_projects_internal,
#     filtered_projects_external,
#     los_df,
#     pdb="full_p2cq_pub_2024-12-03.csdsql",
#     rf_home=".",
# ):
#     query_project = project[identifier]
#     if query_project != query_project:
#         internal_reader = io.MoleculeReader(
#             str(Path(rf_home) / "full_p2cq_roche_2024-12-03.csdsql")
#         )
#         pub_mol2_files = Path("/home/tosstora/scratch/LoS/protonate3d_2024-12-03/PPMOL2_SYM/").glob("*_out/*_protonate3d_relabel.mol2")
#         pub_mol2_files = [str(mf) for mf in pub_mol2_files]
#         # pdb_reader = io.MoleculeReader(str(Path(rf_home) / pdb))
#         pdb_reader = io.MoleculeReader(pub_mol2_files)

#         if len(identifier.split("_")[0]) == 4:
#             query_protein = pdb_reader.molecule(identifier)
#         elif len(identifier.split("_")[0]) == 5:
#             query_protein = internal_reader.molecule(identifier)
#         query_protein.remove_hydrogens()
#         query_protein = utils.assign_index_to_atom_label(query_protein)

#         query_central_atom_indices = (
#             los_df[los_df["molecule_name"] == query_protein.identifier][
#                 "query_atom_id"
#             ].unique()
#             - 1
#         )
#         query_central_atom_labels = [
#             query_protein.atoms[query_central_atom_index].label
#             for query_central_atom_index in query_central_atom_indices
#         ]

#         try:
#             internal_entries = filtered_projects_internal[query_project]
#         except KeyError:
#             internal_entries = []
#         for e in internal_entries:
#             reference_protein = internal_reader.molecule(e)
#             reference_protein.remove_hydrogens()
#             if _mcs_includes_central_atom(
#                 query_protein, reference_protein, query_central_atom_labels
#             ):
#                 return True

#         try:
#             external_entries = filtered_projects_external[query_project]
#         except KeyError:
#             external_entries = []
#         for e in external_entries:
#             reference_protein = pdb_reader.molecule(e)
#             reference_protein.remove_hydrogens()
#             reference_protein = utils.assign_index_to_atom_label(reference_protein)
#             if _mcs_includes_central_atom(
#                 query_protein, reference_protein, query_central_atom_labels
#             ):
#                 return True
#     else:
#         return False


def molecule_is_redundant(
    los_df,
    query_ligand,
    non_redundant_project_molecules,
    mode,
    query_central_atom_labels=None,
):
    similarity_searcher = search.SimilaritySearch(query_ligand, threshold=0.5)
    entry_los_df = los_df[los_df["molecule_name"] == query_ligand.identifier]
    for non_redundant_molecule in non_redundant_project_molecules:
        similarity = similarity_searcher.search_molecule(
            non_redundant_molecule
        ).similarity
        if similarity > 0.5:
            mcs = _mcs_includes_central_atom(
                query_ligand,
                non_redundant_molecule,
                mode,
                query_central_atom_labels,
                entry_los_df,
            )
            if mcs:
                return mcs, non_redundant_molecule.identifier
        else:
            continue
    return False


def find_additional_mcs_matches(
    internal_ligands, external_ligands, mcs, clusters, additional_mcs_matches
):
    substructure = search.QuerySubstructure()
    atom_dict = {}
    mcs, cluster_identifier = mcs
    for atom in mcs[1]:
        atom_dict[atom.label] = substructure.add_atom(atom)

    for bond in mcs[0]:
        substructure.add_bond(
            bond, atom_dict[bond.atoms[0].label], atom_dict[bond.atoms[1].label]
        )
    for m in internal_ligands + external_ligands:
        identifier = m.identifier
        if (
            identifier in list(clusters.keys())
            or identifier in clusters[cluster_identifier]
            or identifier in additional_mcs_matches
        ):
            continue
        if substructure.match_molecule(m):
            additional_mcs_matches.append(identifier)
            clusters[cluster_identifier].append(identifier)
        else:
            continue
    return additional_mcs_matches


def return_ligands(molecule_list):
    ligands = []
    for m in molecule_list:
        ligand = [c for c in m.components if "_Z" in c.atoms[0].label][0]
        ligand.identifier = m.identifier
        ligands.append(ligand)
    return ligands


def remove_mcs_redundancies(
    los_df,
    complex_df,
    filtered_projects_internal,
    filtered_projects_external,
    rf_home="/LoS/protonate3d_2024-12-03-12",
    mode="ligand",
    output="mcs_clusters.csv",
    pub_db="full_p2cq_pub_2024-12-03",
    roche_db=None,
):

    internal_reader_path = Path(rf_home) / f"{roche_db}.csdsql"
    if internal_reader_path.is_file():
        internal_reader = io.MoleculeReader(str(internal_reader_path))

    # pub_mol2_files = Path(
    #     "/home/tosstora/scratch/LoS/protonate3d_2024-12-03/PPMOL2_SYM/"
    # ).glob("????_out/????_???_protonate3d_relabel.mol2")
    # pub_mol2_files = [str(mf) for mf in pub_mol2_files]
    # pdb_reader = io.MoleculeReader(pub_mol2_files)
    pdb_reader = io.MoleculeReader(str(Path(rf_home) / f"{pub_db}.csdsql"))

    clusters_df = (
        pd.DataFrame()
    )  # for logging purposes, contains clustered entry identifiers.

    non_redundant_molecules = []

    # sort projects for reproducibility
    los_df = los_df.sort_values(by="molecule_name")
    complex_df = complex_df.sort_values(by="molecule_name")
    filtered_identifiers = complex_df["molecule_name"].unique()

    for project in sorted(
        set(
            list(filtered_projects_external.keys())
            + list(filtered_projects_internal.keys())
        )
    ):
        t1 = time.time()
        print(project)
        if project != project:
            continue

        internal_molecules = []
        external_molecules = []

        if project in filtered_projects_internal.keys() and internal_reader:
            internal_molecules = [
                internal_reader.molecule(m)
                for m in filtered_projects_internal[project]
                if m in filtered_identifiers
            ]
        if project in filtered_projects_external.keys():
            external_molecules = [
                pdb_reader.molecule(m)
                for m in filtered_projects_external[project]
                if m in filtered_identifiers
            ]
        internal_ligands = return_ligands(internal_molecules)
        external_ligands = return_ligands(external_molecules)
        non_redundant_project_ligands = []
        additional_mcs_matches = []
        clusters = {}

        for query_protein in internal_molecules + external_molecules:
            if query_protein.identifier in additional_mcs_matches:
                continue
            query_protein.remove_hydrogens()
            query_protein = utils.assign_index_to_atom_label(query_protein)
            if mode == "ligand":
                query_central_atom_indices = (
                    los_df[
                        (los_df["molecule_name"] == query_protein.identifier)
                        & (los_df["res_name"] != "SOL")
                    ]["query_atom_id"].unique()
                    - 1
                )

                query_central_atom_labels = [
                    query_protein.atoms[query_central_atom_index].label
                    for query_central_atom_index in query_central_atom_indices
                ]
            else:
                query_central_atom_labels = None
            query_ligand = [
                c for c in query_protein.components if "_Z" in c.atoms[0].label
            ][0]
            query_ligand.identifier = query_protein.identifier
            query_ligand.remove_hydrogens()

            mcs = molecule_is_redundant(
                los_df,
                query_ligand,
                non_redundant_project_ligands,
                mode,
                query_central_atom_labels,
            )
            if mcs:
                additional_mcs_matches = find_additional_mcs_matches(
                    internal_ligands,
                    external_ligands,
                    mcs,
                    clusters,
                    additional_mcs_matches,
                )

                continue
            else:
                clusters[query_ligand.identifier] = []
                non_redundant_project_ligands.append(query_ligand)
        for cluster in clusters:
            clusters_df = pd.concat(
                [
                    clusters_df,
                    pd.DataFrame(
                        {
                            "identifier": clusters[cluster],
                            "cluster_rep": [cluster for i in clusters[cluster]],
                            "project": [project for i in clusters[cluster]],
                        }
                    ),
                ]
            )
        print(project, " took: ", time.time() - t1)

        non_redundant_molecules = (
            non_redundant_molecules + non_redundant_project_ligands
        )

    non_redundant_identifiers = [m.identifier for m in non_redundant_molecules]

    los_mcs_filtered_df = los_df[
        los_df["molecule_name"].isin(non_redundant_identifiers)
    ]
    complex_mcs_filtered_df = complex_df[
        complex_df["molecule_name"].isin(non_redundant_identifiers)
    ]
    complex_mcs_filtered_df = complex_mcs_filtered_df.drop(columns="ROMol")
    clusters_df.to_csv(output, index=False)

    return los_mcs_filtered_df, complex_mcs_filtered_df


def is_glycol(smiles: str) -> bool:
    """
    Check whether a ligand is a glycol or not.
    :param smiles:
    :return: True if ligand is a glycol, False if not.
    >>> smiles = '[O][C][C]O[C][C]O[C][C]O[C][C]O[C][C]O[C][C]O[C][C][O]'
    >>> is_glycol(smiles)
    True
    >>> smiles = '[O][C][C]1O[C](O[C]2[C]([O])[C]([O])[C]([O])O[C]2[C][O])[C]([O])[C]([O])[C]1[O]'
    >>> is_glycol(smiles)
    False
    """
    glycol_substructure = (
        "[OD1,OD2$([O][CD2][CD2][O])][CD2][CD2][OD1,OD2$([O][CD2][CD2][O])]"
    )
    glycol_substructure = Chem.MolFromSmarts(glycol_substructure)
    ligand = Chem.MolFromSmiles(smiles)
    if ligand is None:
        return False
    matches = ligand.GetSubstructMatches(glycol_substructure, uniquify=True)
    # flatten list
    matched_atoms = set(list(chain.from_iterable(matches)))
    if len(matched_atoms) == ligand.GetNumAtoms():
        return True
    else:
        return False


def is_long_chain(complex_df):
    for i, smiles in complex_df["ligand_smiles"].items():
        try:
            rdkit_mol = Chem.MolFromSmiles(smiles)
            complex_df.loc[i, "ROMol"] = rdkit_mol
        except:
            continue
    complex_df = complex_df[complex_df["ROMol"].isna() == False]

    tmp_ligand_file = tempfile.NamedTemporaryFile(suffix=".sdf").name
    PandasTools.WriteSDF(
        complex_df[complex_df["ROMol"].isna() == False],
        tmp_ligand_file,
        idName="molecule_name",
    )

    csd_ligands = io.MoleculeReader(tmp_ligand_file)

    long_chain_ids = []
    for i in reversed(range(3, 50)):
        smarts = "[*X1]" + "".join(["[*X2]" for chain_length in range(i)]) + "[*X1]"
        searcher = search.SubstructureSearch()
        substructure = search.SMARTSSubstructure(smarts)
        searcher.add_substructure(substructure)
        hits = searcher.search(csd_ligands, max_hits_per_structure=1)
        for hit in hits:
            hit_match_atoms = len(hit.match_atoms())
            if (
                hit_match_atoms > 30
                or hit_match_atoms / len(hit.molecule.heavy_atoms) >= 0.9
            ):
                long_chain_ids.append(hit.identifier)
    complex_df = complex_df.drop(columns="ROMol")
    return list(set(long_chain_ids))


class P2cqFilter(object):
    def __init__(
        self,
        input,
        resolution,
        rscc,
        structure_quality_file,
        mode,
        filter_out_cofactor=True,
        occu=1,
        rf_home=".",
        pub_db="full_p2cq_pub_2024-12-03.csdsql",
        roche_db=None,
    ):
        self.input = input
        self.rf_home = rf_home
        self.resolution = float(resolution)
        self.rscc = float(rscc)
        self.occu = float(occu)

        self.struc_qual_df = pd.read_parquet(
            Path(self.rf_home) / structure_quality_file,
        )
        self.struc_qual_df = self.struc_qual_df.rename(columns={"bs_id": "identifier"})
        self.struc_qual = structure_quality.StructureQuality(self.struc_qual_df)

        self.filter_out_cofactor = filter_out_cofactor
        if self.filter_out_cofactor:
            print("Filtering out cofactors.")
        if not self.filter_out_cofactor:
            print("Not filtering out cofactors.")
        self.cofactor_list = protein.Protein.known_cofactor_codes() + [
            "AMP",
            "ADP",
            "ATP",
            "GMP",
            "GDP",
            "GTP",
        ]
        self.mode = mode
        self.annotation_df = self.struc_qual_df
        self.output_extension = "_filtered.csv"

        atom_type_path = Path(atom_types.__path__[0])
        self.protein_atom_types = pd.read_csv(
            atom_type_path / "protein_atom_types.csv", sep="\t"
        )
        self.pub_db = pub_db
        self.roche_db = roche_db

    def merge_and_keep_largest_alpha_i(self, los_files):
        """

        :param los_files: List of los csv files
        :return: los csv file with duplicates filtered by smallest angle
        """
        combined_los = pd.concat(
            [
                pd.read_csv(f"{los_file}{self.output_extension}")
                for los_file in los_files
            ]
        )
        combined_los = combined_los.sort_values(
            ["res_name", "molecule_name", "distance", "alpha_i"]
        )
        combined_los.to_csv(f"los{self.output_extension}", index=False)
        return combined_los

    def filter_files(self):
        files = list(Path(self.input).glob("./*.csv"))
        complex_file = [
            file
            for file in files
            if "complex" in file.name and "filtered" not in file.name
        ][0]
        complex_df = pd.read_csv(Path(complex_file))
        complex_df["strucid"] = (
            complex_df["molecule_name"].str.split("_", expand=True)[0].str.lower()
        )
        los_file = [
            file for file in files if "los" in file.name and "filtered" not in file.name
        ][0]

        # filter by X-ray quality
        complex_df = complex_df[
            complex_df["molecule_name"].isin(self.struc_qual.low_quality_bs_ids)
            == False
        ]
        complex_df = complex_df[
            complex_df["strucid"].isin(self.struc_qual.low_quality_strucids) == False
        ]

        # select complex with maximal contact surface area for given strucid:smiles combination
        if self.mode == "ligand":
            tot_surf_columns = ["surf_area_other_ligand", "surf_area_metal"] + [
                f"surf_area_{pat}"
                for pat in self.protein_atom_types["protein_atom_type"].unique()
                if pat != "Water"
            ]
            tot_surf_columns = [c for c in tot_surf_columns if c in complex_df.columns]
            complex_df["total_surface"] = complex_df.loc[:, tot_surf_columns].sum(
                axis=1
            )
        elif self.mode == "protein":
            complex_df["total_surface"] = (
                complex_df["surf_area_protein"] + complex_df["surf_area_other_ligand"]
            )
        complex_df = complex_df.sort_values(
            "total_surface", ascending=False
        ).drop_duplicates(["strucid", "ligand_smiles"])

        # assign project id, if no project, then use UNIPROT
        complex_df = complex_df.drop(columns=["PROA_PROJ", "PROJECT"], errors="ignore")
        complex_df = complex_df.join(
            self.annotation_df[
                [
                    "identifier",
                    "PROJECT",
                    "uniprot_id",
                    "MinCorr",
                    "MeanCorr",
                    "RESOLUTION",
                    "ligand_name",
                ]
            ]
            .drop_duplicates("identifier")
            .set_index("identifier"),
            on="molecule_name",
        )

        # exclude cofactors
        if self.filter_out_cofactor:
            complex_df = complex_df[
                complex_df["ligand_name"]
                .str.strip("[]'")
                .str[:3]
                .isin(self.cofactor_list)
                == False
            ]
        complex_df.loc[
            (complex_df["PROJECT"] == "_UNCLASSIFIED")
            | (complex_df["PROJECT"].isna())
            | (complex_df["PROJECT"] == "misc_complexes"),
            "PROJECT",
        ] = complex_df["uniprot_id"]
        complex_df["project_smiles"] = (
            complex_df["PROJECT"] + ":" + complex_df["ligand_smiles"]
        )
        if "ligand_rscc" not in complex_df.columns:
            complex_df = complex_df.join(
                self.struc_qual_df[["identifier", "ligand_rscc"]]
                .drop_duplicates()
                .set_index("identifier"),
                on="molecule_name",
            )
        public_complex_df = complex_df[complex_df["strucid"].str.len() == 4]
        public_complex_df = public_complex_df.sort_values(
            ["ligand_rscc", "total_surface"], ascending=[False, False]
        ).drop_duplicates("project_smiles")
        internal_complex_df = complex_df[complex_df["strucid"].str.len() == 5]
        internal_complex_df = internal_complex_df.sort_values(
            ["RESOLUTION", "MeanCorr", "total_surface"], ascending=[True, False, False]
        ).drop_duplicates("project_smiles")
        complex_df = pd.concat([internal_complex_df, public_complex_df])
        complex_df = complex_df.sort_values(
            ["RESOLUTION", "total_surface"], ascending=[True, False]
        ).drop_duplicates("project_smiles")

        long_chain_ids = is_long_chain(complex_df)
        complex_df = complex_df[
            complex_df["molecule_name"].isin(long_chain_ids) == False
        ]

        self.annotation_df = self.annotation_df[
            self.annotation_df["strucid"].isin(
                [i.lower().split("_")[0] for i in complex_df["molecule_name"].unique()]
            )
        ]
        self.annotation_df = self.annotation_df.drop(
            columns=["RESOLUTION", "uniprot_id"]
        )

        los_df = pd.read_csv(Path(los_file))
        los_df = los_df[los_df["molecule_name"].isin(complex_df["molecule_name"])]

        # project = {}
        # strucid = {}
        # lines = {}
        # hitlist = []
        # count = 0

        # for index, line in complex_df.iterrows():
        #     count += 1
        #     id = line["molecule_name"]
        #     strucid[id] = id.split("_")[0]
        #     lines[id] = lineF

        #     try:
        #         project_id = line["PROJECT"]

        #         if project_id in ["_UNCLASSIFIED", "misc_complexes", np.nan]:
        #             project[id] = line["uniprot_id"]
        #         else:
        #             project[id] = project_id
        #     except IndexError:
        #         project[id] = line["uniprot_id"]
        #     hitlist.append(id)

        # complex_df = complex_df[complex_df["molecule_name"].isin(hitlist)]
        # los_df = los_df[los_df["molecule_name"].isin(hitlist)]

        print("Filter out redundant scaffolds...")
        complex_df = complex_df[
            (complex_df["PROJECT"].isna() == False)
            | (complex_df["uniprot_id"].isna() == False)
        ]
        complex_df.loc[complex_df["PROJECT"].isna(), "PROJECT"] = complex_df[
            "uniprot_id"
        ]
        filtered_projects_internal = {
            proj: complex_df[
                (complex_df["PROJECT"] == proj) & (complex_df["strucid"].str.len() == 5)
            ]["molecule_name"].to_list()
            for proj in complex_df[complex_df["strucid"].str.len() == 5]["PROJECT"]
            if proj == proj
        }

        filtered_projects_external = {
            proj: complex_df[
                (complex_df["PROJECT"] == proj) & (complex_df["strucid"].str.len() == 4)
            ]["molecule_name"].to_list()
            for proj in complex_df[complex_df["strucid"].str.len() == 4]["PROJECT"]
            if proj == proj
        }

        los_df = los_df.astype({"query_atom_id": int})
        los_df, complex_df = remove_mcs_redundancies(
            los_df,
            complex_df,
            filtered_projects_internal,
            filtered_projects_external,
            rf_home=self.rf_home,
            mode=self.mode,
            output=self.input / "mcs_clusters.csv",
            pub_db=self.pub_db,
            roche_db=self.roche_db,
        )

        los_df.to_csv(self.input / "los_filtered.csv", index=False)
        complex_df.to_csv(self.input / "complex_filtered.csv", index=False)
