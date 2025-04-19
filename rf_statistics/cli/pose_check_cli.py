from ccdc import io, entry
from rf_statistics import los_descriptors
from pathlib import Path
import click


def _return_describer(**kwargs):
    if kwargs["gold_conf"]:
        describer = los_descriptors.CsdDescriptorsFromGold(
            kwargs["ligand_file"],
            gold_conf=kwargs["gold_conf"],
            only_binding_site=False,
        )

        return describer, describer.contact_df()

    if kwargs["protein"]:
        describer = los_descriptors.CsdDescriptorsFromPDB(
            kwargs["protein"],
            kwargs["ccdc_molecule"],
            only_binding_site=False,
        )

        return describer, describer.los_contacts_df


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "-l",
    "--ligands",
    help="SDF with one ligand structure.",
    required=False,
)
@click.option(
    "-g",
    "--gold_conf",
    help="Path to gold.conf file. Specify gold.conf or protein apo file.",
    default=None,
    required=False,
    type=str,
)
@click.option(
    "-p",
    "--protein",
    help="Path to protein apo file. Specify gold.conf or protein apo file.",
    default=None,
    required=False,
    type=str,
)
@click.option(
    "-o",
    "--output",
    help="Path to output file.",
    default="pose_check.sdf",
    required=False,
)
def pose_check(output: str, ligands: str, protein: str, gold_conf: str):
    if not ligands:
        ligand_files = [str(l) for l in Path(gold_conf).parent.glob("gold_soln*.sdf")]
    else:
        ligand_files = [ligands]
    contact_dfs = []
    output_entries = []

    for ligand_file in ligand_files:
        for ccdc_entry in io.EntryReader(ligand_file):
            ccdc_molecule = ccdc_entry.molecule
            ccdc_molecule.remove_unknown_atoms()
            _e = entry.Entry.from_molecule(ccdc_molecule)
            _e.attributes = ccdc_entry.attributes
            ccdc_entry = _e

            # add RF Histogram
            describer, contact_df = _return_describer(
                gold_conf=gold_conf,
                protein=protein,
                ligand_file=ligand_file,
                ccdc_molecule=ccdc_molecule,
            )

            contact_df = los_descriptors.return_primary_contacts_df(
                contact_df, describer.protein
            )
            contact_dfs.append(contact_df)
            rf_count_df = los_descriptors.rf_count_df(contact_df, ccdc_entry.molecule)
            rf_count_df = rf_count_df.iloc[:, 20:].drop(columns="smiles")
            rf_count_df.columns = [str(c) for c in rf_count_df.columns]
            rf_count_df = rf_count_df.rename({"clash_count": "P-L_steric_clashes"})
            ccdc_entry.attributes.update(rf_count_df.iloc[0].to_dict())

            # RF atom type coverage
            ccdc_entry.attributes["RF_atom_type_coverage"] = describer.ligand_df.shape[
                0
            ] / len(ccdc_molecule.heavy_atoms)
            # AtomRFs
            atom_rf_count_df = los_descriptors.get_atom_rf_count_df(contact_df)
            ccdc_entry.attributes.update(atom_rf_count_df.iloc[0].to_dict())

            # add clash counts
            # clash_dict = pose_processor.count_bad_contacts(ccdc_entry.molecule)
            # ccdc_entry.attributes.update(clash_dict)

            if gold_conf:
                ccdc_entry.attributes["Gold.PLP.Fitness_efficiency"] = float(
                    ccdc_entry.attributes["Gold.PLP.Fitness"]
                ) / len(ccdc_molecule.heavy_atoms)

            # torsion check
            # torsion_df = los_descriptors.torsion_df(ligand_file)
            # ccdc_entry.attributes.update(torsion_df.to_dict())
            output_entries.append(ccdc_entry)

    with io.EntryWriter(output) as w:
        for ccdc_entry in output_entries:
            w.write(ccdc_entry)
        # pd.concat(contact_dfs, ignore_index=True).to_parquet(
        #     output.replace(".sdf", ".gzip")
        # )


if __name__ == "__main__":
    pose_check()
