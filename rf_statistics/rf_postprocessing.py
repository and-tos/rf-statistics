########################################################################################################################

import __future__
import pandas as pd
from pathlib import Path
import numpy as np

from rf_statistics import los_analysis, p2cq_filter, rf_plotter
from rf_statistics import atom_types

########################################################################################################################


def distance_dependency(
    second_geometry_name,
    mode,
    rf_home,
    ligand_atom_type,
    bin_size=0.5,
    bins=[],
    los_input="los_filtered.csv",
    complex_input="complex_filtered.csv",
    outdir=".",
):
    """
    Wrapper that calculates Rf values for attractive and repulsive ranges. Assign unique ligand, protein and binding
     site counts.
    :return: csv file with angle ranges for each atom type, their Rf value with 95% confidence and unique ligand,
     protein and binding site counts.
    """
    if not list(bins):
        bins = np.arange(0, 4.0, bin_size)

    statistics_df = pd.DataFrame()
    for min_dist in bins:
        max_dist = min_dist + bin_size
        output = f"statistics_{second_geometry_name}_{bin_size}_{str(max_dist)}.csv"
        if mode == "protein":
            protein_atom_types = "query_match"
        else:
            protein_atom_types = None
        rf_analyzer = los_analysis.RfAtom(
            second_geometry_min=min_dist,
            second_geometry_max=max_dist,
            second_geometry_name=second_geometry_name,
            output_path=output,
            n_boot=500,
            mode=mode,
            protein_atom_types=protein_atom_types,
            no_export=True,
            rf_home=rf_home,
            los_input=los_input,
            complex_input=complex_input,
        )
        statistics_df = pd.concat([statistics_df, rf_analyzer.calculate_rf()])
    statistics_df = statistics_df.assign(ligand_atom_type=ligand_atom_type)
    outfile = outdir / f"statistics_{second_geometry_name}.csv"
    statistics_df.to_csv(outfile, index=False)


class Postprocessing(object):
    def __init__(
        self,
        db,
        input_folder="rf_statistics",
        rf_home="",
        structure_quality_file="structure_quality.csv",
        angle=False,
        watmap=False,
        generate_watermaps=False,
        two_dimensional=False,
        second_geometry_name=None,
        mode="ligand",
        angle_name="alpha",
        traj=False,
        ligand_atom_type="",
    ):
        self.input = Path(input_folder) / ligand_atom_type
        self.rf_home = rf_home
        self.db = db
        self.structure_quality = structure_quality_file
        self.angle = angle
        self.watmap = watmap
        self.generate_watermaps = generate_watermaps
        self.two_dim = two_dimensional
        self.second_geometry_name = second_geometry_name

        atom_type_path = Path(atom_types.__path__[0])
        self.protein_atom_types = pd.read_csv(
            atom_type_path / "protein_atom_types.csv", sep="\t"
        )
        self.protein_atom_types = list(
            self.protein_atom_types["protein_atom_type"].unique()
        ) + ["other_ligand", "metal"]

        self.mode = mode

        atom_type_path = Path(atom_types.__path__[0])
        ligand_atom_types_df = pd.read_csv(
            atom_type_path / "ligand_atom_types.csv", sep="\t"
        )

        self.ligand_atom_type = ligand_atom_type
        if traj:
            self.rdkit_smarts = ""
            self.rdkit_smarts_index = 0

        else:
            self.rdkit_smarts = ligand_atom_types_df[
                ligand_atom_types_df["ligand_atom_type"] == self.ligand_atom_type
            ]["RDKit_SMARTS"].to_numpy()[0]
            self.rdkit_smarts_index = ligand_atom_types_df[
                ligand_atom_types_df["ligand_atom_type"] == self.ligand_atom_type
            ]["RDKit_SMARTS_index"].to_numpy()[0]

        self.angle_name = angle_name

    def call_p2cq_filter(self, input_folder, resolution_thr=2.5, rscc_thr=0.8):
        """
        Execute p2cq_filter.
        :param resolution_thr: Consider only structures with a resolution <= 2.5.
        :param rscc_thr: Consider only structures with an RSCC >= 0.8.
        :return: Calls p2cq_filter.main()
        """
        filter = p2cq_filter.P2cqFilter(
            input=input_folder,
            resolution=resolution_thr,
            rscc=rscc_thr,
            structure_quality_file=self.structure_quality,
            filter_out_cofactor=True,
            mode=self.mode,
            rf_home=self.rf_home,
            pub_db=[db for db in self.db if "roche" not in db][0],
            roche_db=[db for db in self.db if "roche" in db][0],
        )
        filter.filter_files()

    def angle_dependency(
        self,
        n_boot=500,
        los_input="los_filtered.csv",
        complex_input="complex_filtered.csv",
        outdir=".",
    ):
        """
        Wrapper that calculates Rf values for attractive and repulsive ranges. Assign unique ligand, protein and binding
         site counts.
        :return: csv file with angle ranges for each atom type, their Rf value with 95% confidence and unique ligand,
         protein and binding site counts.
        """
        statistics_df = pd.DataFrame()
        for min_alpha_i in range(0, 180, 20):
            max_alpha_i = min_alpha_i + 20
            output = "".join(
                [f"statistics_{self.angle_name}_", str(180 - min_alpha_i), ".csv"]
            )
            if self.mode == "protein":
                protein_atom_types = "query_match"
            else:
                protein_atom_types = None
            rf_analyzer = los_analysis.RfAtom(
                min_alpha_i=min_alpha_i,
                max_alpha_i=max_alpha_i,
                output_path=output,
                n_boot=n_boot,
                mode=self.mode,
                protein_atom_types=protein_atom_types,
                no_export=True,
                rf_home=self.rf_home,
                los_input=los_input,
                complex_input=complex_input,
                angle_name=self.angle_name,
            )
            statistics_df = pd.concat([statistics_df, rf_analyzer.calculate_rf()])
        statistics_df = statistics_df.assign(
            RDKit_SMARTS=self.rdkit_smarts,
            RDKit_SMARTS_index=self.rdkit_smarts_index,
            ligand_atom_type=self.ligand_atom_type,
        )
        outfile = Path(outdir) / f"statistics_{self.angle_name}.csv"
        statistics_df.to_csv(outfile, index=False)

    def alpha_max(self, rf_angle_range_df, rf_values_p):
        """
        Get the alpha_i with maximum Rf value for a given range. Defined as midpoint between the two alpha values with
         highest Rf. alpha_max is appended to the input DataFrame and the updated DataFrame is returned.
        :param rf_angle_range_df:
        :param rf_values_p:
        :return: DataFrame with 'max_rf_alpha_i' column.
        """
        rf_values_p = pd.read_csv(rf_values_p)
        for index, row in rf_angle_range_df.iterrows():
            if row["type"] != "+":
                continue
            alpha_i_max = int(row["alpha_i_max"])
            alpha_i_min = int(row["alpha_i_min"])
            protein_atom_type = row["atom_type"]
            full_row = rf_values_p[rf_values_p["atom_type"] == protein_atom_type]
            rf_in_range = full_row.loc[
                :, str(180 - alpha_i_min) : str(180 - alpha_i_max + 10)
            ]
            max_rf_alpha = rf_in_range.idxmax(axis=1).values[0]
            max_rf_alpha_index = full_row.columns.get_loc(max_rf_alpha)
            max_rf_alpha_neighbour_1 = full_row.iloc[:, max_rf_alpha_index - 1].name

            # catch error if at limit of Data Frame
            try:
                max_rf_alpha_neighbour_2 = full_row.iloc[:, max_rf_alpha_index + 1].name
            except IndexError:
                alpha_i_max = 180 - 5
                rf_angle_range_df.loc[index, "max_rf_alpha_i"] = alpha_i_max
                continue

            # Calculate midpoint
            if (
                full_row[max_rf_alpha_neighbour_1].values[0]
                >= full_row[max_rf_alpha_neighbour_2].values[0]
            ):
                alpha_max = (
                    float(max_rf_alpha) + float(full_row[max_rf_alpha_neighbour_1].name)
                ) / 2
            else:
                alpha_max = (
                    float(max_rf_alpha) + float(full_row[max_rf_alpha_neighbour_2].name)
                ) / 2

            rf_angle_range_df.loc[index, "max_rf_alpha_i"] = 180 - alpha_max
        return rf_angle_range_df

    def alpha_min(self, rf_angle_range_df, rf_values_p):
        rf_values_p = pd.read_csv(rf_values_p)
        for index, row in rf_angle_range_df.iterrows():
            if row["type"] != "-":
                continue
            alpha_i_max = int(row["alpha_i_max"])
            alpha_i_min = int(row["alpha_i_min"])
            protein_atom_type = row["atom_type"]
            full_row = rf_values_p[rf_values_p["atom_type"] == protein_atom_type]
            rf_in_range = full_row.loc[
                :, str(180 - alpha_i_min) : str(180 - alpha_i_max)
            ]
            min_rf_alpha = rf_in_range.idxmin(axis=1).values[0]
            min_rf_alpha_index = full_row.columns.get_loc(min_rf_alpha)
            min_rf_alpha_neighbour_1 = full_row.iloc[:, min_rf_alpha_index - 1].name
            min_rf_alpha_neighbour_2 = full_row.iloc[:, min_rf_alpha_index + 1].name

            # Calculate midpoint
            if (
                full_row[min_rf_alpha_neighbour_1].values[0]
                <= full_row[min_rf_alpha_neighbour_2].values[0]
            ):
                alpha_min = (
                    float(min_rf_alpha) + float(full_row[min_rf_alpha_neighbour_1].name)
                ) / 2
            else:
                alpha_min = (
                    float(min_rf_alpha) + float(full_row[min_rf_alpha_neighbour_2].name)
                ) / 2

            rf_angle_range_df.loc[index, "alpha_i_min"] = 180 - alpha_min
        return rf_angle_range_df

    def two_dimensional_angle_angle_rf(self, second_geometry_name, protein_atom_types):
        """
        Calculate Rf values as a function of alpha and a second angle.
        :param second_geometry_name: Name of the second geometry
        :param protein_atom_types: Restrain Rf calculation to a list of protein atom types to reduce calculation times.
        :return:
        """
        for min_angle_2 in range(0, 180, 10):
            max_angle_2 = min_angle_2 + 10
            for min_alpha_i in range(0, 180, 10):
                max_alpha_i = min_alpha_i + 10
                output = f"statistics_{str(180 - min_alpha_i)}_{second_geometry_name}_{min_angle_2}_{max_angle_2}.csv"
                rf_analyzer = los_analysis.RfAtom(
                    min_alpha_i=min_alpha_i,
                    max_alpha_i=max_alpha_i,
                    output_path=output,
                    n_boot=500,
                    second_geometry_name=second_geometry_name,
                    second_geometry_max=max_angle_2,
                    second_geometry_min=min_angle_2,
                    protein_atom_types=protein_atom_types,
                    mode=self.mode,
                )
                rf_analyzer.calculate_rf()

    def two_dimensional_angle_distance_rf(self, second_geometry_name):
        """
        Calculate Rf values as a function of alpha and a distance. Distance range is 0 to 4 Angstrom with bin size 0.5.
        :param second_geometry_name: Name of the second geometry
        :param protein_atom_types: Restrain Rf calculation to a list of protein atom types to reduce calculation times.
        :return:
        """
        statistics_df = pd.DataFrame()
        if self.mode == "protein":
            protein_atom_types = "query_match"
        else:
            protein_atom_types = None
        for min_distance in np.arange(0, 5, 0.25):
            max_distance = min_distance + 0.25
            for min_alpha_i in range(0, 180, 10):
                max_alpha_i = min_alpha_i + 10
                output = f"statistics_{str(180 - min_alpha_i)}_{second_geometry_name}_{min_distance}_{max_distance}.csv"
                rf_analyzer = los_analysis.RfAtom(
                    min_alpha_i=min_alpha_i,
                    max_alpha_i=max_alpha_i,
                    output_path=output,
                    n_boot=500,
                    second_geometry_name=second_geometry_name,
                    second_geometry_max=max_distance,
                    second_geometry_min=min_distance,
                    protein_atom_types=protein_atom_types,
                    mode=self.mode,
                    no_export=True,
                )
                statistics_df = statistics_df.append(rf_analyzer.calculate_rf())
        statistics_df.to_csv("statistics_alpha_h.csv", index=False)

    def calculate_preferred_angle_rf(self, interaction_type="attractive", n_boot=500):
        """
        Calculate attractive or repulsive angle ranges.
        :param interaction_type: 'attractive' or 'repulsive'
        :param n_boot: Number of bootstrapping cycles.
        :return: DataFrame with angle ranges.
        """
        preferred_angles = pd.read_csv(f"pref_angle_range_{interaction_type}.csv")
        preferred_dfs = [pd.DataFrame()]
        for row in preferred_angles.iterrows():
            row = row[1]
            pat = row["contact_type"]
            for pref_range in range(1, 5):
                max_alpha_i = 180 - row[f"alpha_range_{pref_range}_min"]
                min_alpha_i = 180 - row[f"alpha_range_{pref_range}_max"]
                if not pd.isnull(min_alpha_i) and not pd.isnull(max_alpha_i):
                    output = f"range_statistics_{pat}_{min_alpha_i}_{max_alpha_i}_.csv"
                    rf_analyzer = los_analysis.RfAtom(
                        min_alpha_i=min_alpha_i,
                        max_alpha_i=max_alpha_i,
                        output_path=output,
                        protein_atom_types=pat,
                        no_export=True,
                        n_boot=n_boot,
                        mode=self.mode,
                    )
                    preferred_df = rf_analyzer.calculate_rf()
                    preferred_df.loc[0, "alpha_i_max"] = max_alpha_i
                    preferred_df.loc[0, "alpha_i_min"] = min_alpha_i
                    preferred_dfs.append(preferred_df)
        rf_angle_range_df = pd.concat(preferred_dfs)
        return rf_angle_range_df

    def plot_protein_angle_ranges(self):
        rf_angle_range_df_with_hitlist_list = []
        for folder in self.protein_atom_types:
            try:
                rf_angle_range_df_with_hitlist = pd.read_csv(
                    Path(folder) / "rf_angle_range_df_with_hitlist.csv"
                )
            except FileNotFoundError:
                rf_angle_range_df_with_hitlist = pd.DataFrame(index=[0])
            if len(rf_angle_range_df_with_hitlist.index) == 0:
                for column in rf_angle_range_df_with_hitlist.columns:
                    rf_angle_range_df_with_hitlist.loc[0, column] = np.nan
            rf_angle_range_df_with_hitlist.loc[:, "atom_type"] = folder
            rf_angle_range_df_with_hitlist_list.append(rf_angle_range_df_with_hitlist)
        try:
            rf_angle_range_df_with_hitlist_combined = pd.concat(
                rf_angle_range_df_with_hitlist_list, sort=False
            )
            rf_plotter.favorable_unfavorable_angle_ranges_heatmap(
                Path("rf_statistics")
                / self.ligand_atom_type
                / f"atom_vs_alpha_rf_ranges_heatmap.png",
                rf_angle_range_df_with_hitlist_combined,
            )
        except ValueError:
            print("No data available, plot cannot be generated.")

    def plot_protein_rf_bars(self, input_path="rf.csv", output_path="protein_rf"):
        rf_df_list = []
        for folder in self.protein_atom_types:
            try:
                rf_df = pd.read_csv(
                    Path("rf_statistics") / self.ligand_atom_type / folder / input_path
                )
                rf_df = rf_df.drop(
                    rf_df[rf_df["atom_type"] == "other_central_ligand"].index
                )
            except FileNotFoundError:
                rf_df = pd.DataFrame(index=[0])
            if len(rf_df.index) == 0:
                for column in rf_df.columns:
                    rf_df.loc[0, column] = np.nan
            rf_df.loc[:, "atom_type"] = folder
            rf_df_list.append(rf_df)
        try:
            protein_rf_df = pd.concat(rf_df_list, sort=False)
            protein_rf_df.to_csv(
                Path("rf_statistics") / self.ligand_atom_type / f"protein_{input_path}",
                index=False,
            )
            rf_plotter.rf_bar_plot(protein_rf_df, "protein_rf", outdir=output_path)
        except ValueError:
            print("No data available, plot cannot be generated.")

        try:
            ligand_rf_df = pd.read_csv(Path("query_atom") / "rf.csv")
            ligand_rf_df.loc[:, "perspective"] = "ligand"
            protein_rf_df.loc[:, "perspective"] = "protein"
            both_rf_df = pd.concat([protein_rf_df, ligand_rf_df])
            both_rf_df = both_rf_df[
                (both_rf_df["atom_type"] != "other_ligand")
                & (both_rf_df["atom_type"] != "don")
                & (both_rf_df["atom_type"] != "pi")
                & (both_rf_df["atom_type"] != "acc")
                & (both_rf_df["atom_type"] != "pos")
                & (both_rf_df["atom_type"] != "neg")
                & (both_rf_df["atom_type"] != "apol")
            ]
            rf_plotter.rf_bar_plot(
                both_rf_df,
                title="rf_protein_ligand_barplot.png",
                hue="perspective",
                outdir=Path("rf_statistics") / self.ligand_atom_type,
            )

        except FileNotFoundError:
            print("Comparison plot not possible.")

    def generate_protein_lookup_file(self, geometry="alpha"):
        protein_lookup = pd.DataFrame()
        bin_dfs = []
        for bin_file in (Path("rf_statistics") / self.ligand_atom_type).glob(
            f"*/statistics_{geometry}.csv"
        ):
            folder = bin_file.parent.name
            if folder in self.protein_atom_types:
                bin_df = pd.read_csv(bin_file)
                bin_df["atom_type"] = folder
                bin_dfs.append(bin_df)
        protein_lookup = pd.concat(bin_dfs, ignore_index=True)
        protein_lookup = protein_lookup.assign(
            SMARTS=self.rdkit_smarts, SMARTS_index=self.rdkit_smarts_index
        )
        protein_lookup.to_csv(
            Path("rf_statistics")
            / self.ligand_atom_type
            / f"protein_lookup_{geometry}.csv",
            index=False,
        )

        print("Rf values were combined")

    def run(self):
        self.calculate_rf(
            angle=self.angle,
            second_geometry_name=self.second_geometry_name,
            watmap=self.watmap,
            two_dim=self.two_dim,
            protein_atom_types=self.protein_atom_types,
        )


class ProteinFilter(Postprocessing):
    def __init__(
        self,
        db,
        input_folder,
        rf_home,
        protein_atom_type,
        structure_quality_file,
        angle=False,
        watmap=False,
        generate_watermaps=False,
        two_dimensional=False,
        ligand_atom_type="",
    ):
        super().__init__(
            db,
            input_folder=input_folder,
            rf_home=rf_home,
            structure_quality_file=structure_quality_file,
            angle=angle,
            watmap=watmap,
            generate_watermaps=generate_watermaps,
            two_dimensional=two_dimensional,
            second_geometry_name="h",
            mode="protein",
            ligand_atom_type=ligand_atom_type,
        )

        self.protein_atom_type = protein_atom_type

    def run(self):
        root_dir = (
            Path("rf_statistics") / self.ligand_atom_type / self.protein_atom_type
        )
        print("Filtering data...")
        self.call_p2cq_filter(root_dir)
        print("Done filtering.")

        if self.protein_atom_type == "C_ali_apol":
            # update entry count in ligand_atom_type.csv
            ligand_atom_type = Path().resolve().parents[0].stem
            ligand_atom_type_file = Path(self.rf_home) / "ligand_atom_types.csv"
            ligand_atom_types = pd.read_csv(ligand_atom_type_file, sep="\t")
            if type(self.db) == list:
                occurrences_label = f"combined_C_ali_apol_occurrences_filtered"
            else:
                dbname = self.db.split(".")[0]
                occurrences_label = f"{dbname}_C_ali_apol_occurrences_filtered"
            occurrences = pd.read_csv(root_dir / "complex_filtered.csv").shape[0]
            ligand_atom_types.loc[
                ligand_atom_types["ligand_atom_type"] == ligand_atom_type,
                occurrences_label,
            ] = occurrences
            ligand_atom_types.to_csv(
                ligand_atom_type_file,
                sep="\t",
                index=False,
            )


class ProteinDistance(Postprocessing):
    def __init__(
        self,
        db,
        input_folder,
        rf_home,
        protein_atom_type,
        structure_quality_file,
        angle=False,
        watmap=False,
        generate_watermaps=False,
        two_dimensional=False,
        geometry_name="h",
        ligand_atom_type="",
    ):
        super().__init__(
            db,
            input_folder=input_folder,
            rf_home=rf_home,
            structure_quality_file=structure_quality_file,
            angle=angle,
            watmap=watmap,
            generate_watermaps=generate_watermaps,
            two_dimensional=two_dimensional,
            second_geometry_name=geometry_name,
            mode="protein",
            ligand_atom_type=ligand_atom_type,
        )

        self.protein_atom_type = protein_atom_type

    def run(self):
        root_dir = Path(self.rf_home) / "rf_statistics" / self.ligand_atom_type
        if self.second_geometry_name == "h" and "pi" not in self.protein_atom_type:
            stat_file = Path(self.protein_atom_type) / "statistics_h.csv"
            if stat_file.is_file():
                stat_file.unlink()

        else:
            try:
                print("Calculating distance dependent Rf...")
                if self.second_geometry_name == "distance":
                    bin_size = 0.25
                    bins = np.arange(2.5, 4.0, bin_size)
                if self.second_geometry_name == "h":
                    bin_size = 0.5
                    bins = np.arange(0, 4.0, bin_size)
                distance_dependency(
                    self.second_geometry_name,
                    self.mode,
                    self.rf_home,
                    self.ligand_atom_type,
                    bins=bins,
                    bin_size=bin_size,
                    outdir=root_dir / self.protein_atom_type,
                    los_input=root_dir / self.protein_atom_type / "los_filtered.csv",
                    complex_input=root_dir
                    / self.protein_atom_type
                    / "complex_filtered.csv",
                )
            except (pd.errors.EmptyDataError, KeyError):
                print("No data available.")

        self.generate_protein_lookup_file(geometry=self.second_geometry_name)
        rf_plotter.plot_protein_geometry_bins(
            input=Path("rf_statistics")
            / self.ligand_atom_type
            / f"protein_lookup_{self.second_geometry_name}.csv",
            geometry=self.second_geometry_name,
            mode=self.mode,
            smarts_index=self.rdkit_smarts_index,
            title=self.rdkit_smarts,
            outdir=Path("rf_statistics") / self.ligand_atom_type,
        )


class ProteinAngle(Postprocessing):
    def __init__(
        self,
        db,
        input_folder,
        rf_home,
        structure_quality_file,
        protein_atom_type,
        ligand_atom_type,
    ):
        """

        :param input_folder:
        :param rf_home:
        :param structure_quality_file:
        :param protein_atom_types:
        """
        super().__init__(
            db=db,
            input_folder=input_folder,
            rf_home=rf_home,
            structure_quality_file=structure_quality_file,
            angle=True,
            mode="protein",
            ligand_atom_type=ligand_atom_type,
        )
        self.protein_atom_type = protein_atom_type

    def run(self):
        root_dir = (
            Path(self.rf_home)
            / "rf_statistics"
            / self.ligand_atom_type
            / self.protein_atom_type
        )
        try:
            protein_perspective_water = False
            if self.protein_atom_type == "Water":
                protein_perspective_water = True
            rf_analyzer = los_analysis.RfAtom(
                root_dir,
                mode=self.mode,
                n_boot=500,
                rf_home=self.rf_home,
                protein_perspective_water=protein_perspective_water,
                output_path=root_dir / "rf.csv",
            )
            rf_analyzer.calculate_rf()
            print("Calculating angle dependent Rf...")
            self.angle_dependency(
                los_input=root_dir / "los_filtered.csv",
                complex_input=root_dir / "complex_filtered.csv",
                outdir=root_dir,
            )
            if self.protein_atom_type == "Water":
                statistics_df = pd.read_csv(root_dir / "statistics_alpha.csv")
                rf_df = pd.read_csv(root_dir / "rf.csv")
                rf_df = rf_df[rf_df["atom_type"] == "query_match"]
                for column in ["rf", "rf_low", "rf_high", "expected", "hits"]:
                    statistics_df[column] = rf_df[column].to_numpy()[0]
                statistics_df.to_csv(
                    root_dir / "statistics_alpha.csv",
                    index=False,
                )
            print("Done calculating and plotting Rf.")

        except (pd.errors.EmptyDataError, KeyError):
            print("No data available.")
        self.plot_protein_rf_bars(
            output_path=Path("rf_statistics") / self.ligand_atom_type
        )
        self.generate_protein_lookup_file(geometry="alpha")
        rf_plotter.plot_protein_geometry_bins(
            input=Path("rf_statistics")
            / self.ligand_atom_type
            / "protein_lookup_alpha.csv",
            mode=self.mode,
            title=self.rdkit_smarts,
            smarts_index=self.rdkit_smarts_index,
            outdir=Path("rf_statistics") / self.ligand_atom_type,
        )
        return


class LigandFilter(Postprocessing):
    def __init__(
        self, db, input_folder, rf_home, structure_quality_file, ligand_atom_type
    ):
        super().__init__(
            db=db,
            input_folder=input_folder,
            rf_home=rf_home,
            structure_quality_file=structure_quality_file,
            ligand_atom_type=ligand_atom_type,
        )

    def run(self):
        root_dir = (
            Path(self.rf_home) / "rf_statistics" / self.ligand_atom_type / "query_atom"
        )
        print("Filtering data...")
        self.call_p2cq_filter(root_dir)
        print("Done filtering.")

        # update entry count in ligand_atom_type.csv
        ligand_atom_type = Path().resolve().parents[0].stem
        if (Path(self.rf_home) / "ligand_atom_types.csv").is_file():
            ligand_atom_types = pd.read_csv(
                Path(self.rf_home) / "ligand_atom_types.csv", sep="\t"
            )
        else:
            atom_type_path = Path(atom_types.__path__[0])
            ligand_atom_types = pd.read_csv(
                atom_type_path / "ligand_atom_types.csv", sep="\t"
            )

        if type(self.db) == list:
            occurrences_label = f"combined_occurrences_filtered"
        else:
            dbname = self.db.split(".")[0]
            occurrences_label = f"{dbname}_occurrences_filtered"
        occurrences = pd.read_csv(root_dir / "complex_filtered.csv").shape[0]
        ligand_atom_types.loc[
            ligand_atom_types["ligand_atom_type"] == ligand_atom_type, occurrences_label
        ] = occurrences
        ligand_atom_types.to_csv(
            Path(self.rf_home) / "ligand_atom_types.csv", sep="\t", index=False
        )


class LigandAngle(Postprocessing):
    def __init__(
        self,
        db,
        input_folder,
        rf_home,
        structure_quality_file,
        angle_name="alpha",
        ligand_atom_type="",
    ):
        super().__init__(
            db=db,
            input_folder=input_folder,
            rf_home=rf_home,
            structure_quality_file=structure_quality_file,
            angle=True,
            mode="ligand",
            angle_name=angle_name,
            ligand_atom_type=ligand_atom_type,
        )

    def run(self):
        root_dir = (
            Path(self.rf_home)
            / "rf_statistics"
            / self.ligand_atom_type
            / Path("query_atom")
        )
        try:
            if self.angle_name == "alpha":  # Calculate geometry independet RF only once
                rf_analyzer = los_analysis.RfAtom(
                    root_dir,
                    mode=self.mode,
                    n_boot=500,
                    rf_home=self.rf_home,
                    output_path=root_dir / "rf.csv",
                )
                rf_analyzer.calculate_rf()
                rf_plotter.rf_bar_plot(
                    root_dir / "rf.csv", "query_atom", outdir=root_dir
                )

            self.angle_dependency(
                los_input=root_dir / "los_filtered.csv",
                complex_input=root_dir / "complex_filtered.csv",
                outdir=root_dir,
            )

            try:
                rf_plotter.plot_protein_geometry_bins(
                    input=root_dir / f"statistics_{self.angle_name}.csv",
                    expected_threshold=10,
                    geometry=self.angle_name,
                    mode=self.mode,
                    title=self.rdkit_smarts,
                    smarts_index=self.rdkit_smarts_index,
                    outdir=root_dir,
                )

                ligand_lookup = pd.read_csv(
                    root_dir / f"statistics_{self.angle_name}.csv"
                )
                ligand_lookup.to_csv(
                    Path(self.rf_home)
                    / "rf_statistics"
                    / self.ligand_atom_type
                    / f"ligand_lookup_{self.angle_name}.csv",
                    index=False,
                )

            except FileNotFoundError:
                print("Plot cannot be generated.")

            print("Done calculating and plotting Rf.")

        except (pd.errors.EmptyDataError, KeyError):
            print("No data available.")
        return


class LigandAngleTraj(Postprocessing):
    def __init__(
        self,
        db,
        input_folder,
        rf_home,
        structure_quality_file,
        angle_name="alpha",
    ):
        super().__init__(
            db=db,
            input_folder=input_folder,
            rf_home=rf_home,
            structure_quality_file=structure_quality_file,
            angle=True,
            mode="ligand",
            angle_name=angle_name,
            traj=True,
        )

    def run(self):
        root_dir = Path(self.input) / "query_atom"
        try:
            # protein atom types
            protein_atom_types_df = pd.read_parquet(
                "/rf_md/2xu4_desmond_md_job_3/rf_stats/traj_rf_only.gzip"
            )
            protein_atom_types_df = protein_atom_types_df[
                protein_atom_types_df["ligand_atom_label"].str.startswith("_Z")
            ]
            protein_atom_types = list(protein_atom_types_df["los_atom_label"].unique())
            if self.angle_name == "alpha":  # Calculate geometry independet RF only once
                rf_analyzer = los_analysis.RfAtom(
                    mode=self.mode,
                    n_boot=500,
                    rf_home=self.rf_home,
                    protein_atom_types=protein_atom_types,
                )
                rf_analyzer.calculate_rf()
                rf_plotter.rf_bar_plot("rf.csv", "query_atom")

            self.angle_dependency()

            try:
                rf_plotter.plot_protein_geometry_bins(
                    input=f"statistics_{self.angle_name}.csv",
                    expected_threshold=10,
                    geometry=self.angle_name,
                    mode=self.mode,
                    title=self.rdkit_smarts,
                    smarts_index=self.rdkit_smarts_index,
                )

                ligand_lookup = pd.read_csv(f"statistics_{self.angle_name}.csv")
                ligand_lookup.to_csv(
                    Path(self.rf_home)
                    / "rf_statistics"
                    / self.ligand_atom_type
                    / f"ligand_lookup_{self.angle_name}.csv",
                    index=False,
                )

            except FileNotFoundError:
                print("Plot cannot be generated.")

            print("Done calculating and plotting Rf.")

        except (pd.errors.EmptyDataError, KeyError):
            print("No data available.")


class LigandDistance(Postprocessing):
    def __init__(
        self,
        db,
        input_folder,
        rf_home,
        second_geometry_name,
        structure_quality_file,
        pi_atom=True,
        ligand_atom_type="",
    ):
        self.pi_atom = pi_atom
        super().__init__(
            db,
            input_folder=input_folder,
            rf_home=rf_home,
            structure_quality_file=structure_quality_file,
            angle=False,
            second_geometry_name=second_geometry_name,
            mode="ligand",
            ligand_atom_type=ligand_atom_type,
        )

    def run(self):
        root_dir = (
            Path(self.rf_home) / "rf_statistics" / self.ligand_atom_type / "query_atom"
        )
        print("Calculating distance dependent Rf...")
        if self.second_geometry_name == "distance":
            bins = np.arange(2.5, 4.0, 0.25)
            bin_size = 0.25
        if self.second_geometry_name == "h":
            bins = np.arange(0, 4.0, 0.5)
            bin_size = 0.5

        try:
            distance_dependency(
                self.second_geometry_name,
                self.mode,
                self.rf_home,
                self.ligand_atom_type,
                bins=bins,
                bin_size=bin_size,
                los_input=root_dir / "los_filtered.csv",
                complex_input=root_dir / "complex_filtered.csv",
                outdir=root_dir,
            )
        except KeyError:
            print(f"No geometry with name {self.second_geometry_name}.")

        try:
            rf_plotter.plot_protein_geometry_bins(
                input=root_dir / f"statistics_{self.second_geometry_name}.csv",
                expected_threshold=10,
                geometry=self.second_geometry_name,
                title=self.rdkit_smarts,
                smarts_index=self.rdkit_smarts_index,
                mode=self.mode,
                outdir=root_dir,
            )
            ligand_lookup = pd.read_csv(
                root_dir / f"statistics_{self.second_geometry_name}.csv"
            )
            ligand_lookup = ligand_lookup[ligand_lookup["expected"] > 0]
            ligand_lookup.to_csv(
                Path("rf_statistics")
                / self.ligand_atom_type
                / f"ligand_lookup_{self.second_geometry_name}.csv",
                index=False,
            )

        except FileNotFoundError:
            print("Plot cannot be generated.")
        except (pd.errors.EmptyDataError, KeyError):
            print("No data available.")
        return
