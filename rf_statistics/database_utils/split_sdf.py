#!/usr/bin/env python

import __future__
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, SaltRemover
from rdkit.Chem import rdDistGeom
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from pathlib import Path
import argparse
from pathos.multiprocessing import ProcessingPool
from functools import partial
import itertools

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

########################################################################################################################


def parse_args():
    """Define and parse the arguments to the script."""
    parser = argparse.ArgumentParser(
        description="Split SDF file into N files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # To display default values in help message.
    )

    parser.add_argument(
        "-N", "--mol_num", help="Number of structures per file", type=int, default=5
    )

    parser.add_argument("-i", "--input", help="Input file.", default="file.sdf")

    parser.add_argument("-o", "--output", help="Output base filename.", default="out")

    parser.add_argument(
        "-m", "--minimize", help="Output base filename.", action="store_true"
    )

    parser.add_argument(
        "-es",
        "--enumerate_stereo",
        help="If passed, unassigned stereo centers will be enumerated.",
        choices=["all", "unassigned", "False"],
        default="unassigned",
    )

    parser.add_argument(
        "-np", "--nproc", help="Number of processes.", default=1, type=int
    )

    return parser.parse_args()


def _prepare_molecule(rdkit_mol, args):
    try:
        rdkit_mol.GetProp("_Name")
        remover = SaltRemover.SaltRemover()
        rdkit_mol = remover.StripMol(rdkit_mol)

        if "SRN" in rdkit_mol.GetPropsAsDict(includePrivate=True).keys():
            srn = rdkit_mol.GetProp("SRN")
        else:
            srn = ""
        if "_Name" in rdkit_mol.GetPropsAsDict(includePrivate=True).keys():
            rdkit_mol.SetProp("_Name", rdkit_mol.GetProp("_Name"))

        if args.enumerate_stereo == "all":
            opts = StereoEnumerationOptions(maxIsomers=5, onlyUnassigned=False)
            isomers = tuple(EnumerateStereoisomers(rdkit_mol, options=opts))
            print(f"Enumerating {len(isomers)} stereoisomers...")

        elif args.enumerate_stereo == "unassigned":
            opts = StereoEnumerationOptions(maxIsomers=5, onlyUnassigned=True)
            isomers = tuple(EnumerateStereoisomers(rdkit_mol, options=opts))
            print(f"Enumerating {len(isomers)} stereoisomers...")

        else:
            if args.minimize:
                stereocenters = Chem.FindMolChiralCenters(
                    rdkit_mol, includeUnassigned=True, useLegacyImplementation=False
                )
                unassigned_stereocenters = [s[1] for s in stereocenters if "?" in s[1]]
                if unassigned_stereocenters:
                    print(srn, " has unassigned stereo center.")
                    return
                else:
                    isomers = [rdkit_mol]
            else:
                isomers = [rdkit_mol]
        enumerated_mols = []
        for cnt, isomer in enumerate(isomers):
            isomer.SetProp("_Name", isomer.GetProp("_Name") + f"_{cnt}")
            if args.minimize:
                isomer = Chem.AddHs(
                    isomer, addCoords=True
                )  # add hydrogens to preserve stereo info
                if (
                    rdDistGeom.EmbedMolecule(
                        isomer, useRandomCoords=True, maxAttempts=3
                    )
                    != -1
                ):
                    if AllChem.UFFOptimizeMolecule(isomer):
                        enumerated_mols.append(isomer)
                else:
                    continue
            else:
                enumerated_mols.append(isomer)
        return enumerated_mols
    except KeyboardInterrupt:
        sys.exit()
    except:
        return


def single_process(subset, args):
    prepared_mols = []
    for m in subset:
        prepared_mols.extend(_prepare_molecule(m, args))
    return prepared_mols


def multi_process(nproc, subset, args):
    parallel_prepare_molecule = partial(_prepare_molecule, args=args)
    pool = ProcessingPool(nproc)
    prepared_mols = []
    out = pool.map(parallel_prepare_molecule, subset)
    for o in out:
        if o == o:
            prepared_mols.extend(o)
    return prepared_mols


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


def sanity_check(supplier):
    flag_substructures = ["[S]([F])([F])([F])([F])([F])", "[B]"]
    flag_substructures = [Chem.MolFromSmarts(smarts) for smarts in flag_substructures]
    mols = []
    nMols = len(supplier)
    for i in range(nMols):
        try:
            m = supplier[i]
            substructure_flag = False
            for substructure in flag_substructures:
                if m.HasSubstructMatch(substructure):
                    substructure_flag = True
            if substructure_flag:
                print("Substructure Flag detected")
                continue
        except:
            print("Could not read mol.")
            continue
        if m:
            mols.append(m)
        else:
            print(i)
    return mols


def split_sdf(args):
    input_sdf = args.input
    output = args.output

    if Path(input_sdf).suffix == ".sdf":
        supplier = Chem.rdmolfiles.SDMolSupplier(input_sdf, removeHs=False)

    mols = sanity_check(supplier)

    if args.nproc == 1:
        out_molecules = single_process(mols, args)
    else:
        out_molecules = multi_process(args.nproc, mols, args)

    if args.mol_num > 0:
        outmol_chunks = divide_chunks(out_molecules, args.mol_num)
    else:
        outmol_chunks = [out_molecules]

    for chunk_cnt, outmol_chunk in enumerate(outmol_chunks):
        out_path = Path(output)
        out_path = out_path.parent / Path(out_path.stem + f"_{chunk_cnt+1}.sdf")
        w = Chem.SDWriter(str(out_path))
        w.SetKekulize(False)
        for outmol in outmol_chunk:
            w.write(outmol)
        w.close()


def main():
    args = parse_args()
    split_sdf(args)


if __name__ == "__main__":
    main()
