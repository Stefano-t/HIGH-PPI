import math
import re
import requests
import json
import os

# Download URLs
RCSB_URL = "https://files.rcsb.org/download/{}"
ALPHAFOLD_URL = "https://alphafold.ebi.ac.uk/api/prediction/{}"

# Valid proteins to read from PDB.
MAPPED_PROTEIN = [
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY",
    "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO",
    "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"
]

PROTEIN_2_IDX_TABLE = {
    v: idx for v, idx in zip(MAPPED_PROTEIN, range(len(MAPPED_PROTEIN)))
}


def dist(p1, p2):
    assert len(p1) == len(p2)
    assert len(p1) == 3

    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx**2 + dy**2 + dz**2)


def read_atoms(file, chain=".", model=1):
    _ = model  # @TODO: remove, unused.
    pattern = re.compile(chain)  # @NOTE: 'chain' is never set.

    atoms = []
    ajs   = []
    for line in file:
        line = line.strip()
        if line.startswith("ATOM"):
            type = line[12:16].strip()
            chain = line[21:22]
            if type == "CA" and re.match(pattern, chain):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                ajs_id = line[17:20]
                atoms.append((x, y, z))
                ajs.append(ajs_id)

    return atoms, ajs



def compute_contacts(atoms, threshold):
    contacts = []
    for i in range(len(atoms)-2):
        for j in range(i+2, len(atoms)):
            if dist(atoms[i], atoms[j]) < threshold:
                contacts.append((i+1, j+1))
    return contacts


def pdb_to_cm_and_ajs(file, threshold, chain=".", model=1):
    atoms, ajs = read_atoms(file, chain, model)
    contacts = compute_contacts(atoms, threshold)
    return contacts, ajs


def download_pdb(pdb_name: str, dest_folder: str):
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder, exist_ok=True)
    destination = os.path.join(dest_folder, f"{pdb_name}.pdb")

    # First, try to download from PDB bank.
    url = RCSB_URL.format(pdb_name)
    get = requests.get(url)
    if get.status_code == 200:
        with open(destination, "wb") as f:
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
            with open(destination, "wb") as f:
                f.write(content.content)
            return

    raise RuntimeError("Cannot download {}".format(pdb_name))
