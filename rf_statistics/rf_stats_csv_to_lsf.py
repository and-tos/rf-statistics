#!/usr/bin/env python

########################################################################################################################

import subprocess as sp
import pandas as pd
import argparse
import time
from pathlib import Path
from rf_statistics import atom_types

########################################################################################################################


def submit(output_files, executable):
    # check for duplicates:
    files_with_executable = list(
        sorted(
            [
                file_with_executable
                for file_with_executable in output_files
                if executable in file_with_executable
            ]
        )
    )
    if len(files_with_executable) == 0:
        return True

    if len(files_with_executable) >= 2:
        duplicates = files_with_executable[0:-1]
        print("removing duplicates...")
        for duplicate in duplicates:
            if Path(duplicate).isfile():
                Path(duplicate).unlink()


def submit_to_hpc(output_folder, mode, protein_atom_types=[""], contact_out=None):
    print(mode)
    if mode == "ligand":
        print(output_folder, " ", mode, "contact count...")
        contact_out = (
            sp.check_output(
                ["bsub", "-L", "/bin/sh"],
                stdin=open(Path(output_folder) / f"lsf.sh", "r"),
            )
            .rstrip()
            .decode("utf-8")
            .split("<")[1]
            .split(">")[0]
        )
        print(contact_out)

    for protein_atom_type in protein_atom_types:
        if mode == "protein":
            file_ext = f"_{protein_atom_type}"
        else:
            file_ext = ""
        dependent_lsf_filter = ["bsub", "-w", f"done({contact_out})", "-ti"]
        filter_out = (
            sp.check_output(
                dependent_lsf_filter,
                stdin=open(
                    Path(output_folder) / f"lsf_{mode}_filter{file_ext}.sh", "r"
                ),
            )
            .rstrip()
            .decode("utf-8")
            .split("<")[1]
            .split(">")[0]
        )
        print(filter_out)
    return contact_out


def parse_args():
    """Define and parse the arguments to the script."""
    parser = argparse.ArgumentParser(
        description="Submit jobs to calculate Rf statistics on Basel sHPC cluster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # To display default values in help message.
    )

    parser.add_argument(
        "-np", help="Number of parallel processes for multiprocessing.", default=24
    )

    parser.add_argument("--rf_home", help="Path to LoS home folder.", default=".")
    parser.add_argument(
        "--is_filtered", help="No additional filtering if passed.", action="store_true"
    )

    parser.add_argument(
        "-db",
        "--database",
        nargs="*",
        help="Database name in LoS home folder.",
        default=["pub", "roche"],
    )

    return parser.parse_args()


class RFlsf(object):
    def __init__(self, np, db=["pub", "roche"], rf_home=".", is_filtered=False):
        self.np = np
        self.rf_home = str(Path(rf_home).resolve())
        self.db = " ".join(db)
        atom_type_path = Path(atom_types.__path__[0])
        self.df = pd.read_csv(atom_type_path / "ligand_atom_types.csv", sep="\t")
        self.protein_atom_types_df = pd.read_csv(
            atom_type_path / "protein_atom_types.csv", sep="\t"
        )

        self.protein_atom_types = self.protein_atom_types_df[
            "protein_atom_type"
        ].unique()
        self.code_home = str(Path(__file__).parent)
        self.is_filtered = is_filtered

    def write_lsf_input(self, output_folder, executable, smarts, smarts_index):
        filtered_string = ""
        if self.is_filtered:
            filtered_string = "--is_filtered"
        """

        :param mode: protein or ligand
        :return:
        """
        lsf_string = f"""#!/bin/bash

#BSUB -J atomtyping
#BSUB -n 1
#BSUB -W 12:00
#BSUB -q long
#BSUB -R "rusage[mem=80G/host]"
#BSUB -o {output_folder}/lsf_output/lsf_{executable}_%J.out

conda activate rf-statistics3.3

python {self.code_home}/rf_hpc.py -exe {executable} --smarts \'{smarts}\' --smarts_index {smarts_index} --rf_home {self.rf_home} --database {self.db} {filtered_string}
"""
        lsf_file = open(Path(output_folder) / f"lsf.sh", "w")
        lsf_file.write(lsf_string)

    def write_lsf_postprocessing_input(
        self,
        ligand_atom_type,
        mode,
        executable,
        pi_atom,
    ):
        directory = Path("rf_statistics") / ligand_atom_type
        geometry = executable
        if executable in ["tau", "alpha"]:
            executable = "angle"

        if mode == "protein":
            runtime = "72:00"
            protein_atom_types = self.protein_atom_types

        else:
            runtime = "72:00"
            protein_atom_types = [""]

        for protein_atom_type in protein_atom_types:
            if mode == "protein":
                protein_atom_type_flag = f" --protein_atom_type {protein_atom_type}"
                filename_ext = f"_{protein_atom_type}"
                if "pi" not in protein_atom_type:
                    pi_atom = False
                else:
                    pi_atom = "-pi"
            else:
                protein_atom_type_flag = ""
                filename_ext = ""
            if self.is_filtered:
                filter_str = ""
            else:
                filter_str = f"python {self.code_home}/rf_hpc.py -m {mode} -exe filter --rf_home {self.rf_home}{protein_atom_type_flag} --database {self.db} --ligand_atom_type {ligand_atom_type}"
            alpha_str = f"python {self.code_home}/rf_hpc.py -m {mode} -exe alpha --rf_home {self.rf_home}{protein_atom_type_flag} --database {self.db} --ligand_atom_type {ligand_atom_type}"
            distance_str = f"python {self.code_home}/rf_hpc.py -m {mode} -exe distance --rf_home {self.rf_home}{protein_atom_type_flag} --database {self.db} --ligand_atom_type {ligand_atom_type}"
            h_str = ""
            if pi_atom:
                h_str = f"python {self.code_home}/rf_hpc.py -m {mode} -exe h {pi_atom} --rf_home {self.rf_home}{protein_atom_type_flag} --database {self.db} --ligand_atom_type {ligand_atom_type}"

            lsf_string = f"""#!/bin/bash
#BSUB -J post_{mode[:3]}
#BSUB -n 1
#BSUB -W {runtime}
#BSUB -q long
#BSUB -R "rusage[mem=16G/host]"
#BSUB -o {directory}/lsf_output/lsf_{mode}_{geometry}{filename_ext}_%J.out

conda activate rf-statistics3.3

{filter_str}
{alpha_str}
{distance_str}
{h_str}
"""
            filename = Path(directory) / f"lsf_{mode}_{geometry}{filename_ext}.sh"
            lsf_file = open(filename, "w")
            lsf_file.write(lsf_string)
            lsf_file.close()

    def csv_iterator(self):
        for index, row in self.df.iterrows():
            if type(row["RDKit_SMARTS_index"]) == str:
                smarts_indices = row["RDKit_SMARTS_index"].split(";")
            else:
                smarts_indices = [row["RDKit_SMARTS_index"]]
            ligand_atom_type = row["ligand_atom_type"]
            # if "_amide_" not in ligand_atom_type:
            #     continue

            for smarts_index in smarts_indices:
                smarts = row["RDKit_SMARTS"]
                pi_atom = row["pi_atom"]
                if pi_atom:
                    pi_atom = "-pi"
                else:
                    pi_atom = ""

                output_folder = Path("rf_statistics") / ligand_atom_type

                print(ligand_atom_type)
                Path("rf_statistics").mkdir(exist_ok=True)
                Path(output_folder).mkdir(exist_ok=True)
                (Path(output_folder) / "lsf_output").mkdir(exist_ok=True)

                self.write_lsf_input(
                    output_folder,
                    "contacts",
                    smarts,
                    smarts_index,
                )

                # if "oxygen" in output_folder and "carboxylate" in output_folder:
                #     self.write_lsf_postprocessing_input("ligand", "tau", pi_atom)
                #     ligand_geometries.append("tau")

                self.write_lsf_postprocessing_input(
                    ligand_atom_type,
                    "ligand",
                    "filter",
                    pi_atom,
                )
                self.write_lsf_postprocessing_input(
                    ligand_atom_type,
                    "protein",
                    "filter",
                    pi_atom,
                )

                print("Submitting...")
                contact_out = submit_to_hpc(output_folder, "ligand")
                submit_to_hpc(
                    output_folder, "protein", self.protein_atom_types, contact_out
                )

                time.sleep(1)


def main():
    args = parse_args()
    lsf_submittor = RFlsf(args.np, args.database, args.rf_home, args.is_filtered)
    lsf_submittor.csv_iterator()


if __name__ == "__main__":
    main()
