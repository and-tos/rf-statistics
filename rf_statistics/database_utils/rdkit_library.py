#!/usr/bin/env python

########################################################################################################################

import _pickle as pickle

from rdkit import Chem
from rdkit.Chem import rdSubstructLibrary
import time
from pathlib import Path
import argparse
from ccdc import io
from functools import partial
import multiprocessing, logging
from datetime import datetime

# Pickle molecules contain properties
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

########################################################################################################################


def parse_args():
    """Define and parse the arguments to the script."""
    parser = argparse.ArgumentParser(
        description="""
        RDKit Supplier generator.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # To display default values in help message.
    )

    parser.add_argument(
        "-i", "--input", default="public", choices=["public", "internal"]
    )

    parser.add_argument(
        "-n",
        "--nproc",
        help="Number of parallel processes for multiprocessing.",
        default=24,
        type=int,
    )

    return parser.parse_args()


def create_logger():

    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s| %(levelname)s| %(processName)s] %(message)s"
    )
    handler = logging.FileHandler("rdkit_library.log")
    handler.setFormatter(formatter)

    # this bit will make sure you won't have
    # duplicated messages in the output
    if not len(logger.handlers):
        logger.addHandler(handler)
    return logger


def rdkit_mol_from_ccdc_mol(ccdc_mol):
    logger = create_logger()
    m = Chem.MolFromMolBlock(ccdc_mol.to_string("sdf"), removeHs=False)
    if m is None:
        m = Chem.MolFromMolBlock(
            ccdc_mol.components[0].to_string("sdf"), removeHs=False
        )
        for c in ccdc_mol.components[1:]:
            component = Chem.MolFromMolBlock(c.to_string("sdf"), removeHs=False)
            if component:
                m = Chem.rdmolops.CombineMols(m, component)
            else:
                logger.info(ccdc_mol.identifier)
                return None

    for cnt, a in enumerate(ccdc_mol.atoms):
        m.GetAtomWithIdx(cnt).SetProp("_TriposAtomName", a.label)

    m.SetProp("_Name", ccdc_mol.identifier)
    return m


def get_rdkit_mol(file):
    """
    :param file:
    :return:
    """
    try:
        with io.EntryReader(str(file)) as rdr:
            for e in rdr:
                ccdc_mol = e.molecule
                m = rdkit_mol_from_ccdc_mol(ccdc_mol)

        if type(m) == Chem.rdchem.Mol:
            return m
    except:
        return None


def run_multiprocessing(db, nproc):
    parallel_supplier = partial(get_rdkit_mol)
    pool = multiprocessing.Pool(nproc)
    mols = pool.map(parallel_supplier, db)
    return mols


def singleprocessing(db):
    mols = []
    for file in db:
        new_mol = get_rdkit_mol(file)
        mols.append(new_mol)
    return mols


def get_rdkit_mols(db, nproc=1):
    if nproc == 1:
        mols = singleprocessing(db)
    else:
        mols = run_multiprocessing(db, nproc)
    return mols


def write_rdkit_library(rdkit_mols, library_name):
    print("Generating Library...")
    mol_holder = rdSubstructLibrary.CachedMolHolder()
    for m in rdkit_mols:
        if type(m) == Chem.rdchem.Mol:
            m.UpdatePropertyCache()
            Chem.rdmolops.FastFindRings(m)
            m = Chem.rdmolops.AddHs(m, explicitOnly=True)
            mol_holder.AddMol(m)
    library = rdSubstructLibrary.SubstructLibrary(mol_holder)
    print("Dumping library...")
    with open(library_name, "wb+") as rdkit_library:
        pickle.dump(library, rdkit_library, protocol=4)
    return


def make_library(strucid_length, library_name, nproc=1):

    filestr = (
        "".join(["?" for i in range(strucid_length)])
        + "_out/*_protonate3d_relabel.mol2"
    )
    filestr = r"" + filestr
    print(filestr)
    mol2_files = Path("PPMOL2_SYM").glob(filestr)

    rdkit_mols = get_rdkit_mols(mol2_files, nproc)
    write_rdkit_library(rdkit_mols, library_name)
    return


def main():
    logging.basicConfig(filename="rdkit_library.log", filemode="w", level=logging.INFO)
    args = parse_args()
    date_string = (
        f"{datetime.now().year}-{datetime.now().month:02n}-{datetime.now().day:02n}"
    )
    if args.input == "public":
        library_name = f"full_p2cq_pub_{date_string}_rdkit.p"
        strucid_length = 4
    elif args.input == "internal":
        library_name = f"full_p2cq_roche_{date_string}_rdkit.p"
        strucid_length = 5
    t1 = time.time()
    make_library(strucid_length, library_name, args.nproc)
    t2 = time.time()
    print("That took %.2f seconds." % (t2 - t1))


if __name__ == "__main__":
    main()
