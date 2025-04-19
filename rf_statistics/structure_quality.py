import __future__
import pandas as pd
import numpy as np


def get_low_quality_strucids(df):
    low_q_df = df[
        (df["RESOLUTION"] > 2.5)
        | (df["ligand_rscc"] < 0.8)
        | (df["ligand_avgoccu"] < 1.0)
        | (df["MinCorr"] < 0.7)
        | ((df["ligand_altcode"].isna() == False) & (df["ligand_altcode"] != " "))
    ]
    return list(low_q_df["strucid"].unique()), list(low_q_df["identifier"].unique())


def get_high_quality_strucids(df):
    pdb_highq_df = df[
        (df["ligand_rscc"] >= 0.8)
        & (df["RESOLUTION"] <= 2.5)
        & (df["ligand_avgoccu"] == 1.0)
        & (df["ligand_altcode"] == " ")
    ]

    roche_high_q_df = df[
        (df["RESOLUTION"] <= 2.5) & (df["MeanCorr"] > 0.8) & (df["MinCorr"] > 0.7)
    ]
    high_q_df = pd.concat(
        [pdb_highq_df, roche_high_q_df], ignore_index=True
    ).drop_duplicates()
    return list(high_q_df["strucid"].unique()), list(high_q_df["identifier"].unique())


class StructureQuality(object):
    def __init__(self, df) -> None:

        df[["MinCorr", "MeanCorr"]] = (
            df[["MinCorr", "MeanCorr"]].replace("-", np.nan).astype(float)
        )
        self.low_quality_strucids, self.low_quality_bs_ids = get_low_quality_strucids(
            df
        )

        self.high_quality_strucids, self.high_quality_bs_ids = (
            get_high_quality_strucids(df)
        )
        return
