#!/usr/bin/env python

# This script can be used for any purpose without limitation subject to the
# conditions at http://www.ccdc.cam.ac.uk/Community/Pages/Licences/v2.aspx
#
# This permission notice and the following statement of attribution must be
# included in all copies or substantial portions of this script.
#
# 2019-08-14: created by the Cambridge Crystallographic Data Centre
#
"""
Obtain quality attributes for Proasis entries from PDB and write them to a CSV.
"""

###################################################################################################################

import __future__
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from io import StringIO
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import xml.etree.ElementTree as ET

from rf_statistics.utils import run_multiprocessing

###################################################################################################################


def parse_args():
    """Define and parse the arguments to the script."""
    parser = argparse.ArgumentParser(
        description="""
        Obtain quality attributes for Proasis entries from PDB and write them to a CSV.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # To display default values in help message.
    )

    parser.add_argument(
        "-o",
        "--outname",
        help="Ouput filename.",
        default="full_p2cq_pub_2024-12-03",
    )

    parser.add_argument(
        "-np", "--nproc", type=int, help="Number of processes", default=1
    )

    return parser.parse_args()


def mol2_to_df(mol2_string):
    record_types = mol2_string.split("@<TRIPOS>")
    atoms_string = "\n".join(record_types[2].split("\n")[1:])
    bonds_string = "\n".join(record_types[3].split("\n")[1:])
    mol_string = "\n".join(record_types[1].split("\n")[1:])
    # residue_string = "\n".join(record_types[4].split("\n")[1:])
    for atom_label in [
        "_ZN",
        "_UN",
        "_ZO",
        "_UO",
        "_ZC",
        "_UC",
        "_ZCl",
        "_UCl",
    ]:

        atoms_string = atoms_string.replace(f"{atom_label} ", atom_label)
    atoms_df = pd.read_csv(StringIO(atoms_string), sep=r"\s+", header=None)
    # residue_df = pd.read_csv(
    #     StringIO(residue_string),
    #     sep=r"\s+",
    #     header=None,
    #     names=["resname", "atom_id", "type", 4, "chain", "rescode", 7],
    # )
    return atoms_df, bonds_string, mol_string


def _get_bs_file(bs_files, bs_id):
    for bs_file in bs_files:
        with open(bs_file, "r") as mol2_file:
            mol2_string = mol2_file.read()
            if bs_id in mol2_string:
                return bs_file


def _get_ligname(bs_id):
    strucid = bs_id.split("_")[0].lower()
    bs_files = Path(
        f"/home/tosstora/scratch/LoS/protonate3d_2024-12-03/PPMOL2_SYM/{strucid}_out"
    ).glob("*_temp.mol2")
    filename = _get_bs_file(bs_files, bs_id)
    with open(filename, "r") as mol2_file:
        mol2_string = mol2_file.read()
    atoms_df, bonds_string, mol_string = mol2_to_df(mol2_string)
    lignames = atoms_df[atoms_df[1].str.startswith("_Z")]

    lig_res = lignames[7].str.split("_", expand=True)
    lignames = lig_res[0] + lig_res[1]
    lignames = pd.concat([lignames, lig_res[2]], axis=1)
    lignames = lignames.drop_duplicates()
    return list(lignames.itertuples(index=False))


def url_response(url):
    """
    Getting JSON response from URL
    :param url: String
    :return: JSON
    """
    r = requests.get(url=url)
    # Status code 200 means 'OK'
    if r.status_code == 200:
        json_result = r.json()
        return json_result
    else:
        print(r.status_code, r.reason)
        return None


def run_val_search(pdb_id):
    """
    Check pdbe search api documentation for more detials
    :param pdbe_search_term: String
    :return: JSON
    """
    # This constructs the complete query URL
    base_url = r"https://www.ebi.ac.uk/pdbe/api/"
    validation_url = r"validation/summary_quality_scores/entry/"
    full_query = base_url + validation_url + pdb_id
    val_score = url_response(full_query)
    return val_score


def parse_xml(pdb_id):

    base_url = r"https://www.ebi.ac.uk/pdbe/entry-files/download/"
    validation_url = r"_validation.xml"
    full_query = base_url + pdb_id + validation_url
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    xml = requests.get(full_query, verify=False)
    tree = ET.ElementTree(ET.fromstring(xml.content))
    root = tree.getroot()
    return root


class PdbVal(object):
    def __init__(
        self,
        outname,
        nproc=1,
    ):
        self.outname = outname
        self.nproc = nproc

    def scrap_pdb(self, pdb_id):
        """
        Get structure quality information from PDB.
        :param hit_protein:
        :param query_atom:
        :return: Dictionary with structure quality information.
        """

        pdb_id = pdb_id.lower()
        root = parse_xml(pdb_id)
        overall_quality = run_val_search(pdb_id)[pdb_id]["overall_quality"]
        return root, overall_quality

    def return_pdb_row(self, root, overall_quality, bs_id):
        lignames = _get_ligname(bs_id)
        dic = {
            "identifier": [bs_id],
            "overall_quality": [overall_quality],
            "ligand_rscc": [],
            "ligand_chain_id": [],
            "resolution": [],
            "ligand_avgoccu": [],
            "ligand_altcode": [],
            "ligand_name": [],
        }
        for ligname in lignames:
            lig_quality_dict = self.extract_ligand_quality(root, ligname)
            for key in lig_quality_dict.keys():
                dic[key].append(lig_quality_dict[key])
            dic["ligand_name"].append(ligname[0])

        dic["ligand_name"] = [dic["ligand_name"]]
        dic["ligand_chain_id"] = [dic["ligand_chain_id"]]
        dic["resolution"] = [dic["resolution"][0]]
        dic["ligand_avgoccu"] = [np.median(dic["ligand_avgoccu"])]
        dic["ligand_rscc"] = [np.median(dic["ligand_rscc"])]
        if len(set(dic["ligand_altcode"])) == 1 and dic["ligand_altcode"][0] == " ":
            dic["ligand_altcode"] = [" "]
        else:
            dic["ligand_altcode"] = ["A"]

        row = pd.DataFrame.from_dict(dic)
        return row

    def return_df(self, pdb_id, pdb_dic, intents=5):
        print(pdb_id)
        bs_ids = pdb_dic[pdb_id]
        try:
            for intent in range(intents):
                try:
                    root, overall_quality = self.scrap_pdb(pdb_id)
                    break
                except TimeoutError:
                    continue
            bs_df_list = []
            for bs_id in bs_ids:
                try:
                    row = self.return_pdb_row(root, overall_quality, bs_id)
                    bs_df_list.append(row)
                except Exception as ex:
                    print(ex)
                    row = pd.DataFrame()
                    return row
            bs_df = pd.concat(bs_df_list, ignore_index=True)
            return bs_df

        except Exception as ex:
            print(ex)
            row = pd.DataFrame()
            return row

    def extract_ligand_quality(self, root, ligname):
        resname = ligname[0][0:3]
        chain_label = ligname[1]
        ligand_rscc = np.nan
        ligand_avgoccu = np.nan
        ligand_altcode = np.nan
        resolution = np.nan

        for child in root:
            try:
                resolution = child.attrib["PDB-resolution"]
                break
            except:
                continue

        for child in root:
            if (
                "resname" in child.attrib
                and child.attrib["resname"] == resname
                and "chain" in child.attrib
                and child.attrib["chain"] == chain_label
            ):
                try:
                    ligand_rscc = float(child.attrib["rscc"])
                    ligand_avgoccu = float(child.attrib["avgoccu"])
                    ligand_altcode = child.attrib["altcode"]
                except Exception as ex:
                    pass
                break

        return {
            "ligand_rscc": ligand_rscc,
            "ligand_chain_id": chain_label,
            "resolution": resolution,
            "ligand_avgoccu": ligand_avgoccu,
            "ligand_altcode": ligand_altcode,
        }

    # def run_multiprocessing(self, pdb_dic):
    #     """Multiprocessing of hits"""
    #     parallel_return_df = partial(self.return_df, pdb_dic=pdb_dic)
    #     pool = ProcessingPool(self.nproc)
    #     dfs = pool.map(parallel_return_df, pdb_dic)
    #     return dfs

    def run_single_processing(self, pdb_dic):
        dfs = []
        for cnt, pdb_id in enumerate(pdb_dic):
            dfs.append(self.return_df(pdb_id, pdb_dic))
        return dfs

    def run_pdb_val(self):

        pdb_dic = defaultdict(list)

        pdb_df = pd.read_parquet("full_p2cq_pub_2024-12-03_rf_surface.gzip")
        pdb_df = pdb_df[["molecule_name"]].drop_duplicates("molecule_name")
        pdb_df["strucid"] = (
            pdb_df["molecule_name"].str.lower().str.split("_", expand=True)[0]
        )
        pdb_dic = pdb_df.groupby("strucid")["molecule_name"].apply(list).to_dict()

        if self.nproc > 1:
            dfs = run_multiprocessing(self.nproc, pdb_dic, self.return_df)
        else:
            dfs = self.run_single_processing(pdb_dic)
        dfs = [df for df in dfs if not df.empty]
        print("Concatenating data frames...")
        dtypes = dfs[0].dtypes
        df = pd.concat([df.astype(dtypes) for df in dfs], ignore_index=True)

        print("Writing out CSV file...")
        df.to_csv(f"pdb_rscc_binding_site_{self.outname}.csv", index=False)
        print("Output files have been written.")

    # def _return_pdb_water_df(self, pdb_id, intents=10):
    #     good_water_df = pd.DataFrame()
    #     root = False

    #     for intent in range(intents):
    #         try:
    #             root, overall_quality = self.scrap_pdb(pdb_id)
    #             break
    #         except TimeoutError:
    #             continue
    #         except:
    #             break

    #     if root:
    #         for child in root:
    #             if "resname" in child.attrib and child.attrib["resname"] == "HOH":
    #                 try:
    #                     if (
    #                         float(child.attrib["avgoccu"]) == 1
    #                         and child.attrib["altcode"] == " "
    #                         and float(child.attrib["rscc"]) >= 0.9
    #                     ):
    #                         chain = np.nan
    #                         if "chain" in child.attrib:
    #                             chain = child.attrib["chain"]
    #                         good_water_df = good_water_df.append(
    #                             {
    #                                 "pdb_id": pdb_id,
    #                                 "resname": child.attrib["resname"],
    #                                 "resnum": child.attrib["resnum"],
    #                                 "atomname": child.attrib["resname"]
    #                                 + child.attrib["resnum"],
    #                                 "avgoccu": child.attrib["avgoccu"],
    #                                 "altcode": child.attrib["altcode"],
    #                                 "rscc": child.attrib["rscc"],
    #                                 "chain": chain,
    #                             },
    #                             ignore_index=True,
    #                         )
    #                 except:
    #                     continue
    #     return good_water_df

    # def return_water_quality_df(self, intents=10):

    #     with shelve.open(f"full_p2cq_pub_2024-12-03_rdkit.p") as db:
    #         pdb_ids = set([key.split("_")[0].lower() for key in db.keys()])

    #     def single_processing(pdb_ids):
    #         good_water_df_list = []
    #         for pdb_id in pdb_ids:
    #             good_water_df_list.append(self._return_pdb_water_df(pdb_id))

    #     def multi_processing(pdb_ids):
    #         parallel_return_df = partial(self._return_pdb_water_df)
    #         pool = ProcessingPool(self.nproc)
    #         good_water_df_list = pool.map(parallel_return_df, pdb_ids)
    #         return good_water_df_list

    #     if self.nproc > 1:
    #         good_water_df_list = multi_processing(pdb_ids)
    #     else:
    #         good_water_df_list = single_processing(pdb_ids)
    #     print("Finished scraping PDB.")
    #     good_water_df = pd.concat(good_water_df_list, ignore_index=True)

    #     good_water_df.to_parquet("good_water.gzip", compression="gzip")


def main():
    args = parse_args()
    pdbval = PdbVal(outname=args.outname, nproc=args.nproc)
    pdbval.run_pdb_val()


if __name__ == "__main__":
    main()
