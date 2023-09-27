import numpy as np
import os
from tqdm import tqdm
import math
import argparse
import re
import requests
import json
from multiprocessing.pool import ThreadPool

RCSB_URL = "https://files.rcsb.org/download/{}"
ALPHAFOLD_URL = "https://alphafold.ebi.ac.uk/api/prediction/{}"

parser = argparse.ArgumentParser(description='make_adj_set')
parser.add_argument('--distance', default=None, type=float,
                    help="distance threshold")
args = parser.parse_args()

def dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx**2 + dy**2 + dz**2)


def read_atoms(file, chain=".", model=1):
    _ = model  # @TODO: remove model
    pattern = re.compile(chain)

    # current_model = model
    atoms = []
    for line in file:
        line = line.strip()
        if line.startswith("ATOM"):
            type = line[12:16].strip()
            chain = line[21:22]
            if type == "CA" and re.match(pattern, chain):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atoms.append((x, y, z))
        # elif line.startswith("MODEL"):
        #     current_model = int(line[10:14].strip())
    return atoms


def compute_contacts(atoms, threshold):
    contacts = []
    for i in range(len(atoms)-2):
        for j in range(i+2, len(atoms)):
            if dist(atoms[i], atoms[j]) < threshold:
                contacts.append((i+1, j+1))
    return contacts


def write_output(contacts, file):
    for c in contacts:
        file.write("\t".join(map(str, c))+"\n")


def pdb_to_cm(file, threshold, chain=".", model=1):
    atoms = read_atoms(file, chain, model)
    return compute_contacts(atoms, threshold)


def download_pdb(pdb_name):
    # First, try to download from PDB bank.
    url = RCSB_URL.format(pdb_name)
    get = requests.get(url)
    if get.status_code == 200:
        with open(f"{pdb_name}.pdb", "wb") as f:
            f.write(get.content)
        return
    # Then, try to download from AlphaFold
    url = ALPHAFOLD_URL.format(pdb_name)
    get = requests.get(url)
    if get.status_code == 200:
        content = json.loads(get.content)
        assert len(content) == 1

        pdbUrl = content[0].get("pdbUrl")
        if pdbUrl is None:
            raise RuntimeError("No PDB url in AlphaFold for {}".format(pdb_name))
        # Get the actual PDB.
        content = requests.get(pdbUrl)
        if content.status_code == 200:
            with open(f"{pdb_name}.pdb", "wb") as f:
                f.write(content.content)
            return

    raise RuntimeError("Cannot download {}".format(pdb_name))


with open('ensp_uniprot.txt') as f:
    e = f.read()
    e_sp = e.split('ENSP')

with open('protein.SHS27k.sequences.dictionary.pro3.tsv') as f:
    protein_sequence = f.readlines()

list_all = []
pdbs = []
# Collect all pdb files to download.
for liness1 in tqdm(protein_sequence):
    line1 = liness1.split('\t')
    li = line1[0][10:]
    for i in range(1690):
        e_zj = e_sp[i]
        res = li in e_zj
        if res:
            li2 = e_zj[13:-9]
            pdb_file_name = li2 + '.pdb'
            pdbs.append(li2)
            break


# Worker to parallelize.
def worker_fn(pdb_file_name):
    if not os.path.isfile(pdb_file_name):
        try:
            download_pdb(os.path.splitext(pdb_file_name)[0])
        except Exception:
            return False
    return True


# Parallel download.
print("  [DEBUG] Retrieving {} PDB files".format(len(pdbs)))
with ThreadPool() as pool:
    result = pool.map_async(worker_fn, pdbs)
    result.wait()

# Read all the pdb files and extract interactions.
print("  [DEBUG] Reading concat lists from PDB files")
for pdb in tqdm(pdbs):
    pdb_file_name = pdb + ".pdb"
    contacts = pdb_to_cm(open(pdb_file_name, "r"), args.distance)
    list_all.append(contacts)

list_all = np.array(list_all)
np.save('edge_list_12.npy',list_all)
