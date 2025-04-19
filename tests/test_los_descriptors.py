from pathlib import Path

import pytest
from ccdc import io, protein
from rf_statistics.los_descriptors import (
    RfDescriptors,
    CsdDescriptorsFromGold,
    rf_count_df,
    get_atom_rf_count_df,
    return_primary_contacts_df,
)


@pytest.mark.parametrize(
    "strucid, df_shape",
    [("4mk8", (46, 46))],
)
def test_RfDescriptors(strucid, df_shape):
    protein_file = Path(__file__).parent / "testdata" / f"{strucid}_apo.pdb"
    ligand_file = Path(__file__).parent / "testdata" / f"{strucid}_ligand.sdf"
    csd_protein = protein.Protein.from_file(str(protein_file))
    csd_ligand = io.MoleculeReader(str(ligand_file))[0]
    csd_ligand.normalise_labels()
    csd_protein.normalise_labels()
    csd_protein.add_ligand(csd_ligand)
    csd_ligand = csd_protein.ligands[-1]
    describer = RfDescriptors(csd_protein, csd_ligand)
    assert describer.los_contacts_df.shape == df_shape


# @pytest.mark.parametrize(
#     "ligand_file, gold_conf",
#     [("gold_soln_ligand_m1_1.sdf", "api_gold.conf")],
# )
# def test_RfDescriptorsGold(ligand_file, gold_conf):
#     describer = CsdDescriptorsFromGold(
#         ligand_file,
#         gold_conf,
#         only_binding_site=False,
#     )
#     ccdc_entry = io.EntryReader(ligand_file)[0]
#     contact_df = describer.contact_df()
#     contact_df = return_primary_contacts_df(contact_df, describer.protein)
#     rf_df = rf_count_df(contact_df, ccdc_entry.molecule)
#     rf_df = rf_df.iloc[:, 20:].drop(columns="smiles")
#     rf_df.columns = [str(c) for c in rf_df.columns]
#     rf_df = rf_df.rename({"clash_count": "P-L_steric_clashes"})

#     _atom_rf_count_df = get_atom_rf_count_df(contact_df)
#     assert rf_df.shape[0]


def main():
    test_RfDescriptors("4mk8", (46, 46))
    # test_RfDescriptorsGold("ligand.sdf", "gold_rescoring.conf")


if __name__ == "__main__":
    main()
