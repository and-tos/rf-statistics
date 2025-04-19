import pandas as pd
from pathlib import Path
from rf_statistics import atom_types
from rf_statistics import atom_type_assignment
import numpy as np
import shutil


class LosComplexDf(object):
    def __init__(
        self,
        smarts,
        smarts_index,
        surface_df,
        atom_types_df,
        rf_home,
        protein_atom_types,
        is_filtered,
    ) -> None:
        self.protein_atom_types = protein_atom_types
        self.surface_df = surface_df
        self.atom_types_df = atom_types_df
        self.annotation_df = pd.read_parquet(
            rf_home / "pub_roche_project_annotation_2024-12-03.gz"
        )
        self.is_filtered = is_filtered
        ligand_atom_types_df = pd.read_csv(
            Path(atom_types.__file__).parent / "ligand_atom_types.csv", sep="\t"
        )
        self.ligand_atom_type = ligand_atom_types_df[
            (ligand_atom_types_df["RDKit_SMARTS"] == smarts)
            & (ligand_atom_types_df["RDKit_SMARTS_index"] == smarts_index)
        ]["ligand_atom_type"].values[0]

    def add_alpha_to_los_df(self, hits_df, alphas):
        alpha_labels = [f"alpha_{i}" for i in range(alphas.shape[1])]
        hits_df[alpha_labels] = alphas
        hits_df[alpha_labels] = hits_df[alpha_labels].replace(
            r"^\s*$", np.nan, regex=True
        )
        hits_df[alpha_labels] = hits_df[alpha_labels].astype(float)

        for i in range(alphas.shape[1]):
            hits_df[f"alpha_i_{i}"] = 180 - hits_df[f"alpha_{i}"]
        los_df = (
            180
            - pd.melt(
                hits_df,
                value_vars=alpha_labels,
                ignore_index=False,
                value_name="alpha_i",
            ).dropna()[["alpha_i"]]
        )

        los_df = los_df.join(
            hits_df[[c for c in hits_df.columns if not c.startswith("alpha_")]]
        )
        los_df = los_df.reset_index(drop=True)
        los_df = los_df.drop(
            columns=[
                "ligand_smiles",
            ]
        )
        return los_df

    def make_protein_los_df(self, hits_df):
        hits_df = hits_df.drop(
            columns=[
                c
                for c in hits_df.columns
                if c.endswith(
                    (
                        "_symbol",
                        "_chain",
                        "_is_donor",
                        "_is_acceptor",
                    )
                )
            ]
        )

        hits_df = hits_df.rename(
            columns={
                "query_atom_label": "ligand_query_match_atom_label",
            }
        )
        hits_df = hits_df.rename(
            columns={c: c.replace("los_atom_", "query_atom_") for c in hits_df.columns}
        )
        hits_df = hits_df.rename(
            columns={c: c.replace("ligand_atom_", "los_atom_") for c in hits_df.columns}
        )
        hits_df = hits_df.rename(
            columns={
                "los_atom_residue_label": "res_name",
                "query_atom_index": "query_atom_id",
                "los_atom_index": "los_atom_id",
                "protein_h": "h",
            }
        )
        hits_df = hits_df[hits_df["los_atom_label"].isna() == False]
        hits_df.loc[
            (hits_df["los_atom_label"].str.startswith(("_U", "_Z")) == False),
            "los_atom_type",
        ] = "protein"

        lig_query_atom_label_dict = dict(
            self.atom_types_df.groupby("molecule_name")["query_atom_label"].apply(list)
        )

        query_match_indices = []
        other_central_ligand_indices = []

        for molecule_name, group_df in hits_df.groupby(["molecule_name"]):
            query_atom_labels = lig_query_atom_label_dict[molecule_name[0]]
            query_match_indices.extend(
                group_df[group_df["los_atom_label"].isin(query_atom_labels)].index
            )
            other_central_ligand_indices.extend(
                group_df[
                    (group_df["los_atom_label"].isin(query_atom_labels) == False)
                    & (group_df["los_atom_label"].str.startswith("_Z"))
                ].index
            )

        hits_df.loc[query_match_indices, "los_atom_type"] = "query_match"
        hits_df.loc[other_central_ligand_indices, "los_atom_type"] = (
            "other_central_ligand"
        )

        alphas = hits_df[f"protein_alphas"].str.split(";", expand=True)

        los_df = self.add_alpha_to_los_df(hits_df, alphas)

        water_df = hits_df[hits_df["protein_atom_type"] == "Water"][
            [c for c in hits_df.columns if not c.startswith("alpha_")]
        ]
        los_df = pd.concat([los_df, water_df], ignore_index=True)

        for pat, group_df in los_df.groupby("protein_atom_type"):
            _dir = Path("rf_statistics") / self.ligand_atom_type / Path(pat)
            _dir.mkdir(exist_ok=True)
            if self.is_filtered:
                group_df.to_csv(
                    _dir / f"los_filtered.csv",
                    index=False,
                )
            else:
                group_df.to_csv(
                    _dir / f"{pat}_los.csv",
                    index=False,
                )

        return

    def make_ligand_los_df(self, mode, hits_df):
        hits_df = hits_df.drop(
            columns=[
                c
                for c in hits_df.columns
                if c.endswith(
                    (
                        "_symbol",
                        "_chain",
                        "_is_donor",
                        "_is_acceptor",
                    )
                )
            ]
        )

        hits_df = hits_df.rename(
            columns={
                "los_atom_residue_label": "res_name",
                "protein_atom_type": "los_atom_type",
                "ligand_atom_index": "query_atom_id",
                "ligand_h": "h",
            }
        )
        gdfs = []
        gdf = hits_df.drop_duplicates(["molecule_name", "query_atom_label"])[
            ["molecule_name", "query_atom_label"]
        ]
        for molecule_name, group_df in gdf.groupby("molecule_name"):
            group_df = group_df.reset_index(drop=True).reset_index()
            gdfs.append(group_df)
        gdf = pd.concat(gdfs).rename(columns={"index": "substructure_match"})
        gdf["substructure_match"] = gdf["substructure_match"] + 1
        hits_df = hits_df.join(
            gdf.set_index(["molecule_name", "query_atom_label"]),
            on=["molecule_name", "query_atom_label"],
        )

        alphas = hits_df[f"{mode}_alphas"].str.split(";", expand=True)

        los_df = self.add_alpha_to_los_df(hits_df, alphas)

        _dir = Path("rf_statistics") / self.ligand_atom_type / "query_atom"
        _dir.mkdir(exist_ok=True)
        los_df = los_df.drop(
            columns=[
                "query_atom_label",
                "los_atom_index",
                "ligand_atom_residue_label",
                "ligand_alphas",
                "protein_alphas",
                "ligand_h",
                "protein_h",
                "ligand_atom_surf",
                "los_atom_surf",
                "ligand_surface",
            ],
            errors="ignore",
        )
        if self.is_filtered:
            los_df.to_csv(_dir / "los_filtered.csv", index=False)
        else:
            los_df.to_csv(_dir / "query_atom_los.csv", index=False)

        return

    def _join_annotation(self, complex_df):
        complex_df["strucid"] = (
            complex_df["molecule_name"].str.split("_", expand=True)[0].str.lower()
        )
        complex_df = complex_df.join(
            self.annotation_df.set_index("STRUCID"), on="strucid"
        )
        complex_df = complex_df.rename(
            columns={"RESOLUTION": "resolution", "UNIPROT_ID": "uniprot_id"}
        )
        return complex_df

    def _ligand_complex_df(
        self,
        complex_df,
    ):
        complex_df = complex_df.drop_duplicates("molecule_name")
        binding_site_surface_df = self.surface_df.pivot_table(
            index="molecule_name",
            columns="protein_atom_type",
            values="atom_surface",
            aggfunc="sum",
            fill_value=0,
        )
        binding_site_surface_df = binding_site_surface_df.rename(
            columns={pat: f"surf_area_{pat}" for pat in binding_site_surface_df.columns}
        )
        for pat in self.protein_atom_types:
            if f"surf_area_{pat}" not in binding_site_surface_df.columns:
                binding_site_surface_df[f"surf_area_{pat}"] = 0
        complex_df = complex_df.join(binding_site_surface_df, on="molecule_name")
        _dir = Path("rf_statistics") / self.ligand_atom_type / "query_atom"
        _dir.mkdir(exist_ok=True)
        complex_df = complex_df.drop(
            columns=[
                "query_atom_label",
                "RDKit_SMARTS",
                "los_atom_label",
                "ligand_atom_residue_label",
                "los_atom_residue_label",
                "distance",
                "ligand_atom_surf",
                "los_atom_surf",
                "ligand_surface",
                "is_primary",
                "ligand_rscc",
                "project",
                "ligand_altcode",
                "ligand_avgoccu",
                "is_cofactor",
                "is_glycol",
                "STRUCID",
                "identifier",
            ],
            errors="ignore",
        )
        # complex_df = self._join_annotation(complex_df)
        if self.is_filtered:
            complex_df.to_csv(_dir / "complex_filtered.csv", index=False)
        else:
            complex_df.to_csv(_dir / "query_atom_complex.csv", index=False)
        return

    def _protein_complex_df(self, complex_df):
        complex_dict = {
            "molecule_name": [],
            "surf_area_query_match": [],
            "surf_area_other_central_ligand": [],
            "surf_area_protein": [],
            "surf_area_other_ligand": [],
        }
        query_atom_label_series = self.atom_types_df.groupby("molecule_name")[
            "query_atom_label"
        ].unique()

        for molecule_name, group_df in self.surface_df.groupby("molecule_name"):
            query_atom_labels = query_atom_label_series[molecule_name]

            # surf area for query matches of central ligand atom
            surf_area_query_match = group_df[
                group_df["atom_label"].isin(query_atom_labels)
            ]["atom_surface"].sum()

            # surf area for other central ligand atoms
            surf_area_other_central_ligand = group_df[
                (
                    (group_df["atom_label"].isin(query_atom_labels) == False)
                    & group_df["atom_label"].str.startswith("_Z")
                )
            ]["atom_surface"].sum()

            # surf area for protein atoms
            surf_area_protein = group_df[
                group_df["atom_label"].str.startswith(("_Z", "_U")) == False
            ]["atom_surface"].sum()

            # surf area for het groups that are not central ligand
            surf_area_other_ligand = group_df[
                group_df["atom_label"].str.startswith("_U")
            ]["atom_surface"].sum()

            complex_dict["molecule_name"].append(molecule_name)
            complex_dict["surf_area_query_match"].append(surf_area_query_match)
            complex_dict["surf_area_other_central_ligand"].append(
                surf_area_other_central_ligand
            )
            complex_dict["surf_area_protein"].append(surf_area_protein)
            complex_dict["surf_area_other_ligand"].append(surf_area_other_ligand)

        complex_df = complex_df.drop_duplicates("molecule_name").drop(
            columns=[
                c
                for c in complex_df.columns
                if c.endswith(("_label", "distance", "type"))
                or c.startswith(
                    ("los_atom", "ligand_atom", "ligand_surface" "RDKit", "is_")
                )
            ]
        )
        complex_df = complex_df.join(
            pd.DataFrame(complex_dict).set_index("molecule_name"),
            on="molecule_name",
        )

        # add annotation
        # complex_df = self._join_annotation(complex_df)
        complex_df.columns = [c.lower() for c in complex_df.columns]
        for pat in self.protein_atom_types:
            pat_dir = Path("rf_statistics") / self.ligand_atom_type / pat
            pat_dir.mkdir(exist_ok=True)
            if self.is_filtered:
                complex_df.to_csv(pat_dir / f"complex_filtered.csv", index=False)
            else:
                complex_df.to_csv(pat_dir / f"{pat}_complex.csv", index=False)

        return

    def make_complex_df(self, mode, complex_df):
        complex_df["STRUCID"] = (
            complex_df["molecule_name"].str.split("_", expand=True)[0].str.lower()
        )

        complex_df = complex_df.drop(
            columns=[
                c
                for c in complex_df.columns
                if c.endswith(
                    (
                        "_index",
                        "_symbol",
                        "_alphas",
                        "_h",
                        "_chain",
                        "_is_donor",
                        "_is_acceptor",
                        "_distance",
                        "_atom_type",
                    )
                )
            ]
        )

        if mode == "ligand":
            self._ligand_complex_df(complex_df)
        elif mode == "protein":
            self._protein_complex_df(complex_df)
        return


def get_contacts_df_columns():
    return [
        "identifier",
        "ligand_atom_index",
        "los_atom_index",
        "ligand_atom_label",
        "ligand_atom_symbol",
        "los_atom_label",
        "los_atom_symbol",
        "ligand_atom_residue_label",
        "los_atom_residue_label",
        "los_atom_chain",
        "ligand_atom_chain",
        "ligand_atom_is_acceptor",
        "los_atom_is_acceptor",
        "ligand_atom_is_donor",
        "los_atom_is_donor",
        "ligand_alphas",
        "protein_alphas",
        "ligand_h",
        "protein_h",
        "vdw_distance",
        "distance",
        "ligand_smiles",
        "ligand_atom_type",
        "protein_atom_type",
        "ligand_atom_surf",
        "los_atom_surf",
        "ligand_surface",
        "los_atom_is_warhead",
        "ligand_atom_is_warhead",
    ]


def traj(ligand_atom_label):
    rf_home = Path("../../")
    print(Path(".").resolve())
    contacts_df_columns = get_contacts_df_columns()
    contacts_df = pd.read_parquet(
        "../../traj_rf_only.gzip",
        columns=contacts_df_columns,
    )
    atom_types_df = contacts_df[["identifier"]].copy()
    atom_types_df["query_atom_label"] = ligand_atom_label
    atom_types_df = atom_types_df.rename(columns={"identifier": "molecule_name"})
    # edit contacts_df

    contacts_df = contacts_df.rename(
        columns={"identifier": "molecule_name", "uniprot": "uniprot_id"}
    )
    contacts_df = contacts_df[
        (contacts_df["los_atom_is_warhead"] == False)
        & (contacts_df["ligand_atom_is_warhead"] == False)
    ]
    contacts_df = contacts_df.drop(columns="ligand_atom_type")

    # parse surfaces

    surface_df = pd.read_parquet("../../traj_rf_surface.gzip")
    surface_df.loc[
        surface_df["atom_label"].str.startswith("_Z") == False, "protein_atom_type"
    ] = surface_df[surface_df["atom_label"].str.startswith("_Z") == False]["atom_label"]

    surface_df = surface_df[surface_df["atom_is_warhead"] == False]

    # set protein atom labels to atom types
    protein_atom_types_df = pd.read_parquet(
        "/rf_md/2xu4_desmond_md_job_3/rf_stats/traj_rf_only.gzip"
    )
    protein_atom_types_df = protein_atom_types_df[
        protein_atom_types_df["ligand_atom_label"].str.startswith("_Z")
    ]
    protein_atom_types = list(protein_atom_types_df["los_atom_label"].unique())

    df_maker = LosComplexDf(
        smarts, smarts_index, surface_df, atom_types_df, rf_home, protein_atom_types
    )

    mode = "ligand"
    hit_contacts_df = atom_types_df.join(
        contacts_df.set_index(["molecule_name", "ligand_atom_label"]),
        on=["molecule_name", "query_atom_label"],
        how="inner",
    ).reset_index(drop=True)
    hit_contacts_df["protein_atom_type"] = hit_contacts_df["los_atom_label"]

    df_maker.make_complex_df(mode, hit_contacts_df.copy())
    df_maker.make_ligand_los_df(mode, hit_contacts_df.copy())

    mode = "protein"
    hit_contacts_df = contacts_df[
        contacts_df["molecule_name"].isin(atom_types_df["molecule_name"])
    ].reset_index(drop=True)

    df_maker.make_protein_los_df(hit_contacts_df.copy())
    df_maker.make_complex_df(mode, hit_contacts_df.copy())

    for csv in Path(".").glob("*/*.csv"):
        if "complex" in csv.name:
            shutil.copy(csv, csv.parent / "complex_filtered.csv")
        elif "los" in csv.name:
            shutil.copy(csv, csv.parent / "los_filtered.csv")
    return


def main(smarts, smarts_index, dbs=["roche", "pub"], rf_home=".", is_filtered=False):
    rf_home = Path(rf_home)

    # parse contacts
    print("Loading contacts_db...")
    dbs = [Path(db).stem for db in dbs]
    contacts_dbs = [(rf_home / f"{db}_rf_contacts.gzip") for db in dbs]

    contacts_dbs = [db for db in contacts_dbs if db.is_file()]

    assert contacts_dbs

    contacts_df_columns = get_contacts_df_columns()

    contacts_df = pd.concat(
        [pd.read_parquet(db, columns=contacts_df_columns) for db in contacts_dbs],
        ignore_index=True,
    )

    atom_types_df = atom_type_assignment.main(
        smarts, smarts_index, dbs, rf_home=rf_home
    )

    # edit contacts_df
    contacts_df = contacts_df[
        (contacts_df["identifier"].isin(atom_types_df["molecule_name"]))
    ]
    contacts_df = contacts_df.rename(
        columns={"identifier": "molecule_name", "uniprot": "uniprot_id"}
    )
    contacts_df = contacts_df[
        (contacts_df["los_atom_is_warhead"] == False)
        & (contacts_df["ligand_atom_is_warhead"] == False)
    ]
    contacts_df = contacts_df.drop(columns="ligand_atom_type")

    # parse surfaces
    surface_dbs = [(rf_home / f"{db}_rf_surface.gzip") for db in dbs]
    surface_df = pd.concat(
        [pd.read_parquet(db) for db in surface_dbs], ignore_index=True
    )
    surface_df = surface_df[
        surface_df["molecule_name"].isin(atom_types_df["molecule_name"])
    ]
    surface_df = surface_df[surface_df["atom_is_warhead"] == False]

    protein_atom_types = list(
        pd.read_csv(
            Path(atom_types.__file__).parent / "protein_atom_types.csv", sep="\t"
        )["protein_atom_type"].unique()
    )
    protein_atom_types.extend(["metal", "other_ligand"])
    df_maker = LosComplexDf(
        smarts,
        smarts_index,
        surface_df,
        atom_types_df,
        rf_home,
        protein_atom_types,
        is_filtered,
    )

    mode = "ligand"
    hit_contacts_df = atom_types_df.join(
        contacts_df.set_index(["molecule_name", "ligand_atom_label"]),
        on=["molecule_name", "query_atom_label"],
        how="inner",
    ).reset_index(drop=True)

    df_maker.make_complex_df(mode, hit_contacts_df.copy())
    df_maker.make_ligand_los_df(mode, hit_contacts_df.copy())

    mode = "protein"
    hit_contacts_df = contacts_df[
        contacts_df["molecule_name"].isin(atom_types_df["molecule_name"])
    ].reset_index(drop=True)

    df_maker.make_protein_los_df(hit_contacts_df.copy())
    df_maker.make_complex_df(mode, hit_contacts_df.copy())

    return


if __name__ == "__main__":
    main()
