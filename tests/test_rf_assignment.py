from ccdc import io
from rf_statistics.rf_assignment import RfAssigner

from pathlib import Path
import pandas as pd
import pytest

testdata = Path(__file__).parent / "testdata"


@pytest.mark.parametrize(
    "strucid, df_shape",
    [("4mk8", (46, 54)), ("7l4t", (30, 54))],
)
def test_RfAssigner(strucid, df_shape):
    rf_assignments = []
    ligand_file = str(testdata / f"{strucid}_ligand.sdf")
    apo_protein_file = str(testdata / f"{strucid}_apo.pdb")
    with io.MoleculeReader(ligand_file) as rdr:
        for cnt, ligand in enumerate(rdr):
            assigner = RfAssigner(ligand, target_file=apo_protein_file)
            rf_assignment_df = assigner.rf_assignments
            rf_assignment_df["ligand_id"] = cnt
            rf_assignments.append(assigner.rf_assignments)

    rf_values_df = (
        pd.concat(rf_assignments, ignore_index=True)
        .sort_values(["ligand_atom_label", "los_atom_label"])
        .reset_index(drop=True)
    )
    assert rf_values_df.shape == df_shape


def main():
    for strucid, df_shape in [("4mk8", (46, 45)), ("7l4t", (30, 45))]:
        test_RfAssigner(strucid, df_shape)

    return


if __name__ == "__main__":
    main()
