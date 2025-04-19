import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)

import pickle
import pandas as pd
from rdkit import Chem
import time
from pathlib import Path

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)


def get_hit_identifier_label(rdkit_mol, substructure, smarts_index):
    hits_dict = {"identifier": [], "query_atom_label": []}
    mol_atom_matches = rdkit_mol.GetSubstructMatches(substructure, uniquify=False)
    identifier = rdkit_mol.GetProp("_Name")
    for mol_match_index in mol_atom_matches:
        atom = rdkit_mol.GetAtomWithIdx(mol_match_index[smarts_index])
        atom_label = atom.GetProp("_TriposAtomName")
        if atom_label.startswith("_Z"):
            hits_dict["identifier"].append(identifier)
            hits_dict["query_atom_label"].append(atom_label)

    return hits_dict


class RdkitSSS(object):
    def __init__(self, smarts, smarts_index) -> None:
        substructure = Chem.MolFromSmarts(smarts)
        if substructure:
            try:
                substructure.UpdatePropertyCache()
            except Chem.rdchem.AtomValenceException:
                pass
        else:
            raise Exception("SMARTS error.")
        self.substructure = substructure
        Chem.rdmolops.FastFindRings(substructure)  # this is necessary to identify rings
        self.smarts_index = smarts_index

    def get_hit_dict(self, rdkit_db):
        matched_mol_indices = rdkit_db.GetMatches(
            self.substructure, maxResults=10000000
        )
        output = []
        for match_index in matched_mol_indices:
            matched_mol = rdkit_db.GetMol(match_index)
            output.append(
                get_hit_identifier_label(
                    matched_mol,
                    substructure=self.substructure,
                    smarts_index=self.smarts_index,
                )
            )

        hits_dict = {"identifier": [], "query_atom_label": []}
        for hd in output:
            for key in hits_dict.keys():
                hits_dict[key].extend(hd[key])

        return hits_dict


def main(
    smarts,
    smarts_index,
    dbs=["full_p2cq_roche_2024-12-03", "full_p2cq_pub_2024-12-03"],
    rf_home="/LoS/protonate3d",
):
    hit_dfs = []
    for db in dbs:
        rdkit_db_name = Path(rf_home) / f"{db}_rdkit.p"
        print(rdkit_db_name)
        with open(rdkit_db_name, "rb") as rdkit_db:
            rdkit_db = pickle.load(rdkit_db)
            t1 = time.time()
            _hit_dict = RdkitSSS(smarts, smarts_index).get_hit_dict(rdkit_db)
            t2 = time.time()
            print("SSS: ", t2 - t1)
            if _hit_dict:
                _hit_df = pd.DataFrame(_hit_dict)
                _hit_df["RDKit_SMARTS"] = smarts
                _hit_df["RDKit_SMARTS_index"] = smarts_index
                hit_dfs.append(_hit_df)
            t3 = time.time()
            print("DF concat: ", t3 - t2)
    atom_types_df = pd.concat(hit_dfs, ignore_index=True)
    atom_types_df = atom_types_df.rename(
        columns={"identifier": "molecule_name", "uniprot": "uniprot_id"}
    )
    return atom_types_df


if __name__ == "__main__":
    main()
