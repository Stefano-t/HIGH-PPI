import numpy as np
import os
from itertools import cycle
from tqdm import tqdm
import argparse
from multiprocessing.pool import ThreadPool
import torch
from generate_utils import download_pdb, pdb_to_cm_and_ajs, PROTEIN_2_IDX_TABLE


# Define the arg parser.
parser = argparse.ArgumentParser(description='Create adj list and feature map from proteins in the network.')
parser.add_argument('--distance', default=None, type=float,
                    help="distance threshold")
parser.add_argument("--ensp_uniprot", default="ensp_uniprot.txt", help="File where Ensp to Uniprot translatoin is stored")
parser.add_argument("--protein_seq_file", default="protein.SHS27k.sequences.dictionary.pro3.tsv", help="File where to read protein sequences")
parser.add_argument("--all_for_assign", default="all_assign.txt", help="File where to read 'all_for_assign' variable")
parser.add_argument("--out_adj", default="edge_list_12.npy", help="Output file for the adj list")
parser.add_argument("--out_feat", default="x_list_7.pt", help="Output file for the feature list")
parser.add_argument("--pdb_dest", default="pdbs", help="Destination folder for PDB files")

args = parser.parse_args()

# Translation table from ENSP to Uniprot format.
ENSP_2_UNIPROT = {}

# Parse the ensp_uniprot file and populate the translation table.
with open(args.ensp_uniprot) as f:
    # @TODO: change file format to something more manageable.
    all_file = f.read()  # It's just a single line.
    parts = all_file.split("', '")
    for elt in parts:
        e = f.read()
        elt = elt.replace("'", "")  # Remove sporious quotes.
        elt = elt.split("ENSP")
        assert len(elt) == 2
        # Get mapping
        left, right = elt[1].split("\\t")
        ENSP_2_UNIPROT[left] = right

with open(args.protein_seq_file) as f:
    protein_sequence = f.readlines()

# Collect all pdb files to download.
pdbs = []
for liness1 in tqdm(protein_sequence):
    line1 = liness1.split('\t')
    li    = line1[0][9:]
    li2   = ENSP_2_UNIPROT.get(li)
    if li2 is None:
        raise ValueError(f"Unrecognized ENSP: {li}")
    pdbs.append(li2)


# Worker to parallelize.
def worker_fn(args):
    assert len(args) == 2
    pdb_file_name, dest_folder = args

    pdb_name, ext = os.path.splitext(pdb_file_name)
    if len(ext) == 0:
        ext = ".pdb"
    if not os.path.isfile(os.path.join(dest_folder, pdb_name + ext)):
        try:
            download_pdb(pdb_name, dest_folder)
        except Exception:
            return pdb_file_name, False
    return pdb_file_name, True


# Parallel download.
print("  [DEBUG] Retrieving {} PDB files".format(len(pdbs)))
with ThreadPool() as pool:
    result = pool.imap_unordered(worker_fn, zip(pdbs, cycle([args.pdb_dest])))
    for f_name, status in result:
        if not status:
            print("  [ERROR]: cannot download {}".format(f_name))

# Read all the pdb files and extract interactions.
print("  [DEBUG] Computing concat maps and adjacency lists.")

all_for_assign = np.loadtxt(args.all_for_assign)
assert len(all_for_assign.shape) == 2

adj_list     = []
feature_list = []
for pdb in tqdm(pdbs):
    pdb_file_name = pdb + ".pdb"
    with open(pdb_file_name, "r") as f:
        contacts, xx = pdb_to_cm_and_ajs(f, args.distance)

    adj_list.append(contacts)

    # Prepare the feature.
    x_p = np.zeros((len(xx), all_for_assign.shape[1]))
    for j in range(len(xx)):
        idx = PROTEIN_2_IDX_TABLE.get(xx[j])
        if idx is not None:
            x_p[j] = all_for_assign[idx, :]

    feature_list.append(x_p)


# Save.
adj_list = np.array(adj_list)
np.save(args.out_adj, adj_list)
torch.save(feature_list, args.out_feat)
