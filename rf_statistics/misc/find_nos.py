from ccdc import io, descriptors, search, protein
import pandas as pd
from pathlib import Path
from rdkit import Chem
import gzip
import tempfile

protein_files = set(
    [str(f) for f in Path(".").glob("hlr*.pdb.gz") if len(f.stem) == 12]
)
nos_hits = {"strucid": [], "cys_id": [], "lys_id": [], "distance": []}
cys_sub = search.SMARTSSubstructure("[SD1]")
lys_sub = search.SMARTSSubstructure("[ND1]")
searcher = search.SubstructureSearch()
searcher.add_substructure(cys_sub)
searcher.add_substructure(lys_sub)
searcher.add_distance_constraint(
    "DIST1", (0, 0), (1, 0), (0, 3.5), vdw_corrected=False, type="any"
)
for protein_file in protein_files:
    try:
        with gzip.open(protein_file, "rb") as f:
            pdb_block = f.read()
        rdkit_prot = Chem.MolFromPDBBlock(pdb_block)
        temp_prot_file = tempfile.NamedTemporaryFile(suffix=".pdb")
        Chem.MolToPDBFile(rdkit_prot, temp_prot_file.name)
        m = protein.Protein.from_file(temp_prot_file.name)
        m.identifier = Path(protein_file).stem[3:8]
        hits = searcher.search(m)
        for hit in hits:
            cys_atom = hit.match_atoms()[0]
            close_atom = hit.match_atoms()[1]
            if "LYS" in close_atom.residue_label and "CYS" in cys_atom.residue_label:
                nos_hits["strucid"].append(hit.identifier)
                nos_hits["cys_id"].append(cys_atom.residue_label)
                nos_hits["lys_id"].append(close_atom.residue_label)
                nos_hits["distance"].append(
                    descriptors.MolecularDescriptors.atom_distance(cys_atom, close_atom)
                )
    except:
        continue

df = pd.DataFrame(nos_hits)
df.to_csv("roche_db_nos_hits.csv")
