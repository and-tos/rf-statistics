#!/usr/bin/env python

########################################################################################################################

import __future__
import warnings

warnings.simplefilter("error")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path
import argparse
import pandas as pd
import itertools

from ccdc import io, protein
from rf_statistics.utils import assign_index_to_atom_partial_charge
from rf_statistics import los_descriptors
from rf_statistics.utils import run_multiprocessing

########################################################################################################################


def parse_args():
    """Define and parse the arguments to the script."""
    parser = argparse.ArgumentParser(
        description="""
        Submit Calculate Rf values on Basel HPC
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # To display default values in help message.
    )

    parser.add_argument(
        "-np",
        "--nproc",
        help="Number of parallel processes for multiprocessing.",
        default=24,
    )

    parser.add_argument(
        "-db",
        help="DB internal or public",
        default="internal",
    )

    parser.add_argument(
        "--rf_home",
        default=".",
        help="LoS home directory with lookup files and ligand_atom_types.csv and protein_atom_types.csv",
    )

    return parser.parse_args()


def find_mutation_series():

    platinum_db = pd.read_csv("../mutation_series/platinum_flat_file.csv")
    platinum_db = platinum_db[platinum_db["MUT.IN_BINDING_SITE"] == "YES"]
    platinum_db = platinum_db[
        (platinum_db["MUT.MT_PDB"] != "NO") & (platinum_db["MUT.WT_PDB"] != "NO")
    ]
    platinum_db = platinum_db[(platinum_db["MUT.IS_SINGLE_POINT"] != "NO")]
    platinum_db = platinum_db[(platinum_db["MUT.DISTANCE_TO_LIG"] <= 5)]
    platinum_db = platinum_db[(platinum_db["PROT.RESOLUTION"] <= 2.5)]
    platinum_db.to_csv("../mutation_series/platinum_flat_file_filtered.csv")
    rf_assignments = pd.read_csv("full_p2cq_pub_oct2019_rf.csv")
    rf_assignments = rf_assignments[rf_assignments["rf_total"] <= 0.8]
    rf_assignments = rf_assignments[
        (
            rf_assignments["identifier"]
            .apply(lambda x: x.split("_")[0])
            .isin(platinum_db["MUT.MT_PDB"])
        )
        | (
            rf_assignments["identifier"]
            .apply(lambda x: x.split("_")[0])
            .isin(platinum_db["MUT.WT_PDB"])
        )
    ]
    rf_assignments.to_csv("../mutation_series/mutation_series_entries.csv", index=False)


def find_ligand_series():
    """
    Find entries that are in the BindingDB Protein-Ligand-Validation set.
    :return:
    """

    binding_db = pd.read_csv("../ligand_series/binding_db/binding_db.csv")[
        "Series"
    ].unique()
    binding_db = [series.split(" "[0]) for series in binding_db]
    binding_db = [series for series in binding_db if len(series) > 1]
    binding_db = list(itertools.chain(*binding_db))
    binding_db = [identifier.split("-")[0] for identifier in binding_db]
    rf_assignments = pd.read_csv("full_p2cq_pub_2024-12-03_rf.csv")
    rf_assignments = rf_assignments[rf_assignments["rf_total"] <= 0.8]
    rf_assignments = rf_assignments[
        rf_assignments["identifier"].apply(lambda x: x.split("_")[0]).isin(binding_db)
    ]
    rf_assignments.to_csv("prot_lig_val.csv", index=False)


def export_entry(entry, database, output_folder=".", remove_waters=True):
    if remove_waters:
        output_name = f"{entry}.mol2"
    else:
        output_name = f"{entry}_wet.mol2"
    output_name = Path(output_folder) / output_name

    with io.EntryReader(database) as rdr, io.MoleculeWriter(
        output_name, format="mol2"
    ) as wrt:
        e = rdr.entry(entry)
        p = protein.Protein.from_entry(e)
        if remove_waters:
            p.remove_all_waters()
        p = assign_index_to_atom_partial_charge(p)
        wrt.write(p)


class CsdsqlIterString:
    def __init__(self, rdr):
        self.rdr = rdr
        self.a = 0

    def __iter__(self):
        self.a = 0
        self.entry = self.rdr[self.a].to_string()
        return self

    def __next__(self):
        self.a += 1
        if self.a < len(self.rdr):
            return self.rdr[self.a].to_string()
        else:
            raise StopIteration


class DatabaseUtil(object):
    def __init__(self, rf_home):
        self.rf_home = rf_home
        # self.annotation_df = pd.read_csv(
        #     Path(self.rf_home) / "all_annotated_2024-12-03.csv"
        # )
        self.cofactor_list = protein.Protein.known_cofactor_codes() + [
            "AMP",
            "ADP",
            "ATP",
            "GMP",
            "GDP",
            "GTP",
        ]

    # def return_annotation(self, identifier):
    #     project, resolution, uniprot = np.nan, np.nan, np.nan
    #     id = identifier.split("_")[0].lower()
    #     annotation = self.annotation_df[self.annotation_df["STRUCID"] == id].squeeze()
    #     if len(annotation) > 0:
    #         project = annotation["PROA_PROJ"]
    #         resolution = annotation["RESOLUTION"]
    #         uniprot = annotation["UNIPROT_ID"]

    #     return project, resolution, uniprot

    # def return_struc_qual(self, identifier):
    #     rscc, ligand_avgoccu, ligand_altcode = np.nan, np.nan, np.nan
    #     is_cofactor = False
    #     if len(identifier) == 8:
    #         struc_qual = self.annotation_df[
    #             self.annotation_df["identifier"] == identifier
    #         ].squeeze()
    #         if len(struc_qual) > 0:
    #             rscc = struc_qual["ligand_rscc"]
    #             ligand_avgoccu = struc_qual["ligand_avgoccu"]
    #             ligand_altcode = struc_qual["ligand_altcode"]
    #             ligand_name = struc_qual["ligand_name"][:3]
    #             if ligand_name in self.cofactor_list:
    #                 is_cofactor = True
    #     return rscc, ligand_avgoccu, ligand_altcode, is_cofactor

    # def extend_rf_file(self, db_rf_df):
    #     """
    #     Add additional information to the DataFrame, such as counts of unfavorable and favorable interactions,
    #     Uniprot ID.
    #     :return:
    #     """

    #     # add RSCC, add resolution, Uniprot ID, project

    #     identifier = db_rf_df["identifier"].unique()[0]
    #     rscc, ligand_avgoccu, ligand_altcode, is_cofactor = return_struc_qual(
    #         identifier
    #     )
    #     project, resolution, uniprot = return_annotation(identifier)
    #     db_rf_df["project"] = project
    #     db_rf_df["resolution"] = resolution
    #     db_rf_df["uniprot"] = uniprot
    #     db_rf_df["ligand_rscc"] = rscc
    #     db_rf_df["ligand_altcode"] = ligand_altcode
    #     db_rf_df["ligand_avgoccu"] = ligand_avgoccu
    #     db_rf_df["is_cofactor"] = is_cofactor
    #     ligand_smiles = db_rf_df["ligand_smiles"].values[0]
    #     db_rf_df["is_glycol"] = is_glycol(ligand_smiles)

    #     return db_rf_df

    def rf_df_from_file(
        self,
        protein_file,
        calculate_surface=True,
        intramolecular_contacts=False,
    ):
        p = protein.Protein.from_file(str(protein_file))
        try:
            describer = los_descriptors.CsdDescriptorsFromProasis(
                p,
                calculate_surface=calculate_surface,
                intramolecular_contacts=intramolecular_contacts,
            )
            contact_df = describer.los_contacts_df
            atom_surface_area_df = describer.atom_surface_area_df

        except:
            print("ERROR: ", p.identifier)
            contact_df = pd.DataFrame()
            atom_surface_area_df = pd.DataFrame()

        return contact_df, atom_surface_area_df

    def rf_df_from_string(
        self,
        protein_string,
        calculate_surface=True,
        intramolecular_contacts=False,
    ):
        p = protein.Protein.from_string(str(protein_string))
        try:
            describer = los_descriptors.CsdDescriptorsFromProasis(
                p,
                calculate_surface=calculate_surface,
                intramolecular_contacts=intramolecular_contacts,
            )
            contact_df = describer.los_contacts_df
            atom_surface_area_df = describer.atom_surface_area_df

        except:
            print("ERROR: ", p.identifier)
            contact_df = pd.DataFrame()
            atom_surface_area_df = pd.DataFrame()

        return contact_df, atom_surface_area_df

    def single_processing(self, protein_files):
        output = []
        for protein_file in protein_files:
            output.append(self.rf_df_from_file(protein_file))
            # if entry_index == 10:
            #     break
        return zip(*output)

    # def assign_rf_to_database(self, db="internal", nproc=24):

    #     if db == "public":
    #         strucid_length = 4
    #         output_name = "full_p2cq_pub_2024-12-03"

    #     elif db == "internal":
    #         strucid_length = 5
    #         output_name = "full_p2cq_roche_2024-12-03"

    #     filestr = (
    #         "".join(["?" for i in range(strucid_length)])
    #         + "_out/*_protonate3d_relabel.mol2"
    #     )
    #     filestr = r"" + filestr
    #     print(filestr)
    #     protein_files = Path(
    #         "/home/tosstora/scratch/LoS/protonate3d_2024-12-03/PPMOL2_SYM/"
    #     ).glob(filestr)

    #     if nproc > 1:
    #         contact_dfs, surface_dfs = zip(
    #             *run_multiprocessing(nproc, protein_files, self.rf_df_from_file)
    #         )
    #     else:
    #         contact_dfs, surface_dfs = self.single_processing(protein_files)

    #     contact_df = pd.concat(contact_dfs, ignore_index=True)
    #     surface_df = pd.concat(surface_dfs, ignore_index=True)
    #     contact_df.columns = [str(c) for c in contact_df.columns]
    #     contact_df.to_parquet(f"{output_name}_rf_contacts.gzip", compression="gzip")

    #     surface_df.columns = [str(c) for c in surface_df.columns]
    #     surface_df.to_parquet(f"{output_name}_rf_surface.gzip", compression="gzip")

    def assign_rf_to_database(self, db="internal", nproc=24):

        output_name = Path(db).stem

        it = CsdsqlIterString(io.MoleculeReader(db))

        if nproc > 1:
            contact_dfs, surface_dfs = zip(
                *run_multiprocessing(nproc, it, self.rf_df_from_string)
            )
        else:
            contact_dfs, surface_dfs = self.single_processing(it)

        contact_df = pd.concat(contact_dfs, ignore_index=True)
        surface_df = pd.concat(surface_dfs, ignore_index=True)
        contact_df.columns = [str(c) for c in contact_df.columns]
        contact_df.to_parquet(f"{output_name}_rf_contacts.gzip", compression="gzip")

        surface_df.columns = [str(c) for c in surface_df.columns]
        surface_df.to_parquet(f"{output_name}_rf_surface.gzip", compression="gzip")


def main():
    args = parse_args()
    database_utiliser = DatabaseUtil(rf_home=args.rf_home)
    database_utiliser.assign_rf_to_database(db=args.db, nproc=int(args.nproc))


if __name__ == "__main__":
    main()
