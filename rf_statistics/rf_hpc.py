#!/usr/bin/env python


# A. Tosstorff
# 15-APR-2020

########################################################################################################################
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)

import argparse
import matplotlib

matplotlib.use("Agg")

from rf_statistics import rf_postprocessing, complex_and_los_df

########################################################################################################################


def parse_args():
    """Define and parse the arguments to the script."""
    parser = argparse.ArgumentParser(
        description="""
        Execute Line of sight contact scripts.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # To display default values in help message.
    )

    parser.add_argument(
        "--rf_home",
        help="Path to LoS home folder.",
        default=".",
    )

    parser.add_argument(
        "-db",
        "--database",
        nargs="*",
        help="Database name in LoS home folder.",
        default=["pub", "roche"],
    )

    parser.add_argument(
        "-m", "--mode", help="ligand or protein perspective.", default="ligand"
    )

    parser.add_argument(
        "-a",
        "--annotation",
        help="File containing structure annotation.",
        default="pub_roche_project_annotation_2024-12-03.gz",
    )

    parser.add_argument(
        "-s", "--smarts", help="SMARTS string to define ligand atom type.", default=None
    )

    parser.add_argument(
        "-si",
        "--smarts_index",
        help="Index to define atom in SMARTS string.",
        default=None,
    )

    parser.add_argument(
        "-np", help="Number of parallel processes for multiprocessing.", default=24
    )

    parser.add_argument(
        "-pi", "--pi_atom", help="Central atom is in pi system.", action="store_true"
    )

    parser.add_argument(
        "-exe",
        "--executable",
        choices=[
            "contacts",
            "traj_contacts",
            "alpha",
            "alpha_traj",
            "h",
            "distance",
            "filter",
        ],
        help="With 'contacts', contact counts will be generated. 'angle' and 'h' will execute "
        "postprocessing for the corresponding geometry.",
        default=None,
    )

    parser.add_argument(
        "-lat", "--ligand_atom_type", help="Ligand atom type", default=None
    )

    parser.add_argument(
        "-pat", "--protein_atom_type", help="Protein atom type", default=None
    )
    parser.add_argument(
        "--is_filtered", help="No additional filtering if passed.", action="store_true"
    )

    return parser.parse_args()


class RfWrapper(object):
    def __init__(
        self,
        rf_home,
        database,
        annotation,
        ligand_atom_type,
        pi,
        np,
        executable,
        mode,
        protein_atom_type=None,
        angle_name="alpha",
    ):
        self.rf_home = rf_home
        self.db = database
        self.annotations = annotation
        self.structure_quality = annotation
        self.output = "rf_statistics"

        self.ligand_atom_type = ligand_atom_type
        self.np = np
        self.pi = pi
        if ligand_atom_type:
            if "oxygen" in ligand_atom_type and "carboxylate" in ligand_atom_type:
                self.tau_atom = True
            else:
                self.tau_atom = False
        self.angle_name = angle_name
        self.mode = mode
        self.executable = executable
        self.protein_atom_type = protein_atom_type

        if "contacts" in self.executable:
            self.process = complex_and_los_df

        if self.mode == "ligand":
            if self.executable == "filter":
                self.process = rf_postprocessing.LigandFilter(
                    db=self.db,
                    input_folder=self.output,
                    rf_home=self.rf_home,
                    structure_quality_file=self.structure_quality,
                    ligand_atom_type=self.ligand_atom_type,
                )

            if self.executable in ["alpha", "tau"]:
                self.process = rf_postprocessing.LigandAngle(
                    db=self.db,
                    input_folder=self.output,
                    rf_home=self.rf_home,
                    structure_quality_file=self.structure_quality,
                    angle_name=self.executable,
                    ligand_atom_type=ligand_atom_type,
                )

            if self.executable == ["alpha_traj", "tau_traj"]:
                self.process = rf_postprocessing.LigandAngleTraj(
                    db=self.db,
                    input_folder=self.output,
                    rf_home=self.rf_home,
                    structure_quality_file=self.structure_quality,
                    angle_name=self.executable,
                )

            if self.executable == "h":
                self.process = rf_postprocessing.LigandDistance(
                    db=self.db,
                    input_folder=self.output,
                    rf_home=self.rf_home,
                    structure_quality_file=self.structure_quality,
                    second_geometry_name="h",
                    pi_atom=self.pi,
                    ligand_atom_type=ligand_atom_type,
                )

            if self.executable == "distance":
                self.process = rf_postprocessing.LigandDistance(
                    db=self.db,
                    input_folder=self.output,
                    rf_home=self.rf_home,
                    structure_quality_file=self.structure_quality,
                    second_geometry_name="distance",
                    pi_atom=self.pi,
                    ligand_atom_type=ligand_atom_type,
                )

        if self.mode == "protein":
            if self.executable == "filter":
                self.process = rf_postprocessing.ProteinFilter(
                    db=self.db,
                    input_folder=self.output,
                    rf_home=self.rf_home,
                    structure_quality_file=self.structure_quality,
                    protein_atom_type=protein_atom_type,
                    ligand_atom_type=ligand_atom_type,
                )

            if self.executable in ["alpha", "tau"]:
                self.process = rf_postprocessing.ProteinAngle(
                    db=self.db,
                    input_folder=self.output,
                    rf_home=self.rf_home,
                    structure_quality_file=self.structure_quality,
                    protein_atom_type=protein_atom_type,
                    ligand_atom_type=ligand_atom_type,
                )

            if self.executable == "h":
                self.process = rf_postprocessing.ProteinDistance(
                    db=self.db,
                    input_folder=self.output,
                    rf_home=self.rf_home,
                    structure_quality_file=self.structure_quality,
                    protein_atom_type=protein_atom_type,
                    geometry_name=self.executable,
                    ligand_atom_type=ligand_atom_type,
                )

            if self.executable == "distance":
                self.process = rf_postprocessing.ProteinDistance(
                    db=self.db,
                    input_folder=self.output,
                    rf_home=self.rf_home,
                    structure_quality_file=self.structure_quality,
                    protein_atom_type=protein_atom_type,
                    geometry_name=self.executable,
                    ligand_atom_type=ligand_atom_type,
                )


def main():
    args = parse_args()
    rf_analysis = RfWrapper(
        rf_home=args.rf_home,
        database=args.database,
        annotation=args.annotation,
        ligand_atom_type=args.ligand_atom_type,
        pi=args.pi_atom,
        np=args.np,
        executable=args.executable,
        mode=args.mode,
        protein_atom_type=args.protein_atom_type,
    )
    if rf_analysis.executable == "contacts":
        rf_analysis.process.main(
            args.smarts,
            int(args.smarts_index),
            args.database,
            args.rf_home,
            args.is_filtered,
        )
    elif rf_analysis.executable == "traj_contacts":
        rf_analysis.process.traj(args.smarts)

    else:
        rf_analysis.process.run()


if __name__ == "__main__":
    main()
