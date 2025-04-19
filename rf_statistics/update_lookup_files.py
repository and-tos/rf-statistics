#!/usr/bin/env python

########################################################################################################################

import pandas as pd
import argparse
from pathlib import Path
from rf_statistics import atom_types

########################################################################################################################


def parse_args():
    """Define and parse the arguments to the script."""
    parser = argparse.ArgumentParser(
        description="""
        Update lookup files in rf_home.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # To display default values in help message.
    )

    parser.add_argument(
        "--search_home",
        help="The directory were the RF search was initiated. Contains 'output' directory",
        type=str,
        default=".",
    )

    return parser.parse_args()


def _geometry_test(
    geom_min, geom_max, interaction_type, geometry, df, protein_perspective_water=False
):
    if protein_perspective_water:
        return True
    return (
        df[f"{interaction_type}_{geometry}_max"].to_numpy()[0]
        >= geom_max
        >= df[f"{interaction_type}_{geometry}_min"].to_numpy()[0]
        and df[f"{interaction_type}_{geometry}_min"].to_numpy()[0]
        <= geom_min
        <= df[f"{interaction_type}_{geometry}_max"].to_numpy()[0]
    )


def _return_allowed_interaction_type(
    ligand_interaction_types,
    protein_interaction_types,
    central_atom_interaction_constraints,
    geom_max,
    geom_min,
    geometry,
    central_atom_geometry_allowed_interactions,
    protein_perspective_water=False,
):
    if "hbond_weak" in ligand_interaction_types:
        if (
            "hbond_weak" in protein_interaction_types
            or "hbond_classic" in protein_interaction_types
        ):
            if _geometry_test(
                geom_min,
                geom_max,
                "hbond",
                geometry,
                central_atom_interaction_constraints,
                protein_perspective_water,
            ):
                if (
                    "hbond_acc" in ligand_interaction_types
                    and "hbond_don" in protein_interaction_types
                ):
                    central_atom_geometry_allowed_interactions.append("hbond_weak")
                    if "hbond_classic" in protein_interaction_types:
                        central_atom_geometry_allowed_interactions.append("desolvation")
                elif (
                    "hbond_don" in ligand_interaction_types
                    and "hbond_acc" in protein_interaction_types
                ):
                    central_atom_geometry_allowed_interactions.append("hbond_weak")
                    if "hbond_classic" in protein_interaction_types:
                        central_atom_geometry_allowed_interactions.append("desolvation")

    if "hbond_classic" in ligand_interaction_types:
        if "hbond_classic" in protein_interaction_types:
            if _geometry_test(
                geom_min,
                geom_max,
                "hbond",
                geometry,
                central_atom_interaction_constraints,
                protein_perspective_water,
            ):
                if (
                    "hbond_acc" in ligand_interaction_types
                    and "hbond_don" in protein_interaction_types
                ):
                    central_atom_geometry_allowed_interactions.append("hbond_classic")
                elif (
                    "hbond_don" in ligand_interaction_types
                    and "hbond_acc" in protein_interaction_types
                ):
                    central_atom_geometry_allowed_interactions.append("hbond_classic")
                else:
                    central_atom_geometry_allowed_interactions.append(
                        "electrostatic_repulsion"
                    )

        if "hbond_weak" in protein_interaction_types:
            if _geometry_test(
                geom_min,
                geom_max,
                "hbond",
                geometry,
                central_atom_interaction_constraints,
                protein_perspective_water,
            ):
                if (
                    "hbond_acc" in ligand_interaction_types
                    and "hbond_don" in protein_interaction_types
                ):
                    central_atom_geometry_allowed_interactions.append("hbond_weak")
                    central_atom_geometry_allowed_interactions.append("desolvation")
                elif (
                    "hbond_don" in ligand_interaction_types
                    and "hbond_acc" in protein_interaction_types
                ):
                    central_atom_geometry_allowed_interactions.append("hbond_weak")
                    central_atom_geometry_allowed_interactions.append("desolvation")
                else:
                    central_atom_geometry_allowed_interactions.append(
                        "electrostatic_repulsion"
                    )

    if (
        "multipolar_neg" in ligand_interaction_types
        and "multipolar_pos" in protein_interaction_types
    ):
        if _geometry_test(
            geom_min,
            geom_max,
            "multipolar",
            geometry,
            central_atom_interaction_constraints,
            protein_perspective_water,
        ):
            central_atom_geometry_allowed_interactions.append("multipolar")

    if (
        "multipolar_pos" in ligand_interaction_types
        and "multipolar_neg" in protein_interaction_types
    ):
        if _geometry_test(
            geom_min,
            geom_max,
            "multipolar",
            geometry,
            central_atom_interaction_constraints,
            protein_perspective_water,
        ):
            central_atom_geometry_allowed_interactions.append("multipolar")

    if (
        "halogen_don" in ligand_interaction_types
        and "halogen_acc" in protein_interaction_types
    ):
        if _geometry_test(
            geom_min,
            geom_max,
            "halogen_bond",
            geometry,
            central_atom_interaction_constraints,
            protein_perspective_water,
        ):
            central_atom_geometry_allowed_interactions.append("halogen_bond")

    if (
        "ionic_pos" in ligand_interaction_types
        and "ionic_neg" in protein_interaction_types
    ):
        if _geometry_test(
            geom_min,
            geom_max,
            "ionic",
            geometry,
            central_atom_interaction_constraints,
            protein_perspective_water,
        ):
            central_atom_geometry_allowed_interactions.append("ionic")

    if (
        "ionic_neg" in ligand_interaction_types
        and "ionic_pos" in protein_interaction_types
    ):
        if _geometry_test(
            geom_min,
            geom_max,
            "ionic",
            geometry,
            central_atom_interaction_constraints,
            protein_perspective_water,
        ):
            central_atom_geometry_allowed_interactions.append("ionic")

    if "pi" in ligand_interaction_types and "pi" in protein_interaction_types:
        if _geometry_test(
            geom_min,
            geom_max,
            "pi",
            geometry,
            central_atom_interaction_constraints,
            protein_perspective_water,
        ):
            central_atom_geometry_allowed_interactions.append("pi")

    if (
        "hydrophobic" in ligand_interaction_types
        and "hydrophobic" in protein_interaction_types
    ):
        if _geometry_test(
            geom_min,
            geom_max,
            "hydrophobic",
            geometry,
            central_atom_interaction_constraints,
            protein_perspective_water,
        ):
            central_atom_geometry_allowed_interactions.append("hydrophobic")
        else:
            central_atom_geometry_allowed_interactions.append("desolvation")

    if (
        "hydrophobic" not in ligand_interaction_types
        and "hydrophobic" in protein_interaction_types
    ):
        central_atom_geometry_allowed_interactions.append("desolvation")

    if (
        "hydrophobic" in ligand_interaction_types
        and "hydrophobic" not in protein_interaction_types
    ):
        central_atom_geometry_allowed_interactions.append("desolvation")

    if "hydrophobic" in ligand_interaction_types and (
        "hbond_classic" in protein_interaction_types
        or "multipolar_pos" in protein_interaction_types
        or "multipolar_neg" in protein_interaction_types
        or "ionic_pos" in protein_interaction_types
        or "ionic_neg" in protein_interaction_types
    ):
        central_atom_geometry_allowed_interactions.append("desolvation")

    if "hydrophobic" in protein_interaction_types and (
        "hbond_classic" in ligand_interaction_types
        or "multipolar_pos" in ligand_interaction_types
        or "multipolar_neg" in ligand_interaction_types
        or "ionic_pos" in ligand_interaction_types
        or "ionic_neg" in ligand_interaction_types
    ):
        central_atom_geometry_allowed_interactions.append("desolvation")

    if (
        "ionic_neg" in ligand_interaction_types
        and "ionic_neg" in protein_interaction_types
    ):
        central_atom_geometry_allowed_interactions.append("electrostatic_repulsion")

    if (
        "ionic_pos" in ligand_interaction_types
        and "ionic_pos" in protein_interaction_types
    ):
        central_atom_geometry_allowed_interactions.append("electrostatic_repulsion")

    if (
        "multipolar_pos" in ligand_interaction_types
        and "multipolar_pos" in protein_interaction_types
    ):
        central_atom_geometry_allowed_interactions.append("electrostatic_repulsion")

    if (
        "multipolar_neg" in ligand_interaction_types
        and "multipolar_neg" in protein_interaction_types
    ):
        if _geometry_test(
            geom_min,
            geom_max,
            "multipolar",
            geometry,
            central_atom_interaction_constraints,
            protein_perspective_water,
        ):
            central_atom_geometry_allowed_interactions.append("electrostatic_repulsion")

    if (
        "hbond_classic" in ligand_interaction_types
        and "hbond_acc" in ligand_interaction_types
        and "hbond_classic" in protein_interaction_types
        and "hbond_acc" in protein_interaction_types
    ):
        if _geometry_test(
            geom_min,
            geom_max,
            "hbond",
            geometry,
            central_atom_interaction_constraints,
            protein_perspective_water,
        ):
            central_atom_geometry_allowed_interactions.append("electrostatic_repulsion")

    if (
        "hbond_classic" in ligand_interaction_types
        and "hbond_don" in ligand_interaction_types
        and "hbond_classic" in protein_interaction_types
        and "hbond_don" in protein_interaction_types
    ):
        if _geometry_test(
            geom_min,
            geom_max,
            "hbond",
            geometry,
            central_atom_interaction_constraints,
            protein_perspective_water,
        ):
            central_atom_geometry_allowed_interactions.append("electrostatic_repulsion")

    return list(set(central_atom_geometry_allowed_interactions))


def _assign_interaction_type_to_lookup_files(search_home="."):
    lookup_files = Path(search_home).glob("*lookup*.csv")
    ligand_atom_types_df = pd.read_csv(
        Path(atom_types.__file__).parent / "ligand_atom_types.csv", sep="\t"
    )

    protein_atom_types_df = pd.read_csv(
        Path(atom_types.__file__).parent / "protein_atom_types.csv", sep="\t"
    )
    for lookup_file in lookup_files:
        lookup_file = str(lookup_file)
        if "tau" in lookup_file:
            continue

        lookup_df = pd.read_csv(lookup_file, sep="\t")
        lookup_df = lookup_df[lookup_df["atom_type"] != "other_ligands"]
        interaction_types = []
        # indices = []
        for index, row in lookup_df.iterrows():
            protein_atom_type = row["atom_type"]
            ligand_atom_type = row["ligand_atom_type"]

            if "_h.csv" in lookup_file:
                geom_min = float(row["h_min"])
                geom_max = float(row["h_max"])
                geometry = "h"

            else:
                geom_min = float(row["alpha_min"])
                geom_max = float(row["alpha_max"])
                geometry = "alpha"

            rf = row["rf"]
            central_atom_geometry_allowed_interactions = []
            if (
                pd.isna(rf)
                or protein_atom_type == "metal"
                or protein_atom_type == "other_ligands"
            ):
                interaction_types.append(
                    ";".join(central_atom_geometry_allowed_interactions)
                )
                continue

            ligand_interaction_types = (
                ligand_atom_types_df[
                    ligand_atom_types_df["ligand_atom_type"] == ligand_atom_type
                ]
                .loc[:, "ionic_pos":"hydrophobic"]
                .squeeze()
            )
            ligand_interaction_types = ligand_interaction_types[
                ligand_interaction_types == 1
            ]
            ligand_interaction_types = ligand_interaction_types.keys()

            if protein_atom_type == "other_ligand":
                interaction_types.append(
                    ";".join(central_atom_geometry_allowed_interactions)
                )
                continue
            if protein_atom_type == "S_don":
                protein_atom_type = "S_mix"

            protein_interaction_types = (
                protein_atom_types_df[
                    protein_atom_types_df["protein_atom_type"] == protein_atom_type
                ]
                .loc[:, "ionic_pos":"hydrophobic"]
                .iloc[0]
                .squeeze()
            )
            protein_interaction_types = protein_interaction_types[
                protein_interaction_types == 1
            ]

            protein_perspective_water = False
            if "ligand" in lookup_file:
                central_atom_interaction_constraints = ligand_atom_types_df[
                    ligand_atom_types_df["ligand_atom_type"] == ligand_atom_type
                ].loc[:, f"ionic_{geometry}_max":"hydrophobic_h_min"]
            if "protein" in lookup_file:
                central_atom_interaction_constraints = protein_atom_types_df[
                    protein_atom_types_df["protein_atom_type"] == protein_atom_type
                ].loc[:, f"ionic_{geometry}_max":"hydrophobic_h_min"]
                if protein_atom_type == "Water":
                    protein_perspective_water = True
            if not central_atom_interaction_constraints.empty:
                central_atom_geometry_allowed_interactions = (
                    _return_allowed_interaction_type(
                        ligand_interaction_types,
                        protein_interaction_types,
                        central_atom_interaction_constraints,
                        geom_max,
                        geom_min,
                        geometry,
                        central_atom_geometry_allowed_interactions,
                        protein_perspective_water=protein_perspective_water,
                    )
                )

                interaction_types.append(
                    ";".join(central_atom_geometry_allowed_interactions)
                )
                # indices.append(index)
            else:
                interaction_types.append("")
                # indices.append(index)
        lookup_df["interaction_types"] = interaction_types
        lookup_df.to_csv(lookup_file, sep="\t", index=False)


def main():
    args = parse_args()

    ligand_atom_types_df = pd.read_csv(
        Path(atom_types.__file__).parent / "ligand_atom_types.csv", sep="\t"
    )

    for lookup_type in [
        "ligand_lookup_alpha",
        "ligand_lookup_h",
        "ligand_lookup_distance",
        "protein_lookup_alpha",
        "protein_lookup_h",
        "protein_lookup_distance",
    ]:
        lookup_files = list(Path("rf_statistics").glob(f"*/{lookup_type}.csv"))

        if lookup_type != "ligand_lookup_h":
            ligand_atom_types = ligand_atom_types_df["ligand_atom_type"].to_list()

        else:
            ligand_atom_types = ligand_atom_types_df[
                ligand_atom_types_df["pi_atom"] == True
            ]["ligand_atom_type"].to_list()
        completed_atom_types = [lookup_file.parent.name for lookup_file in lookup_files]
        uncompleted_atom_types = list(
            set(ligand_atom_types) - set(completed_atom_types)
        )
        if uncompleted_atom_types:
            print(
                f"Geometry {lookup_type}. Calculcations not completed for {uncompleted_atom_types}"
            )
        global_lookup_df = pd.concat(
            [pd.read_csv(f, sep=",") for f in lookup_files],
            ignore_index=True,
        )
        global_lookup_df.to_csv(f"global_{lookup_type}.csv", sep="\t", index=False)
    _assign_interaction_type_to_lookup_files(search_home=args.search_home)


if __name__ == "__main__":
    main()
