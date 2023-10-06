import numpy as np
import os
from itertools import cycle
from tqdm import tqdm
from functools import partial
import argparse
from multiprocessing.pool import ThreadPool, Pool
import torch
from generate_utils import download_pdb, pdb_to_cm_and_ajs, PROTEIN_2_IDX_TABLE


def create_arg_parser():
    # Define the arg parser.
    parser = argparse.ArgumentParser(description='Create adj list and feature map from proteins in the network.')
    parser.add_argument('--distance', default=None, type=float,
                        help="distance threshold")
    parser.add_argument("--protein_seq_file", default="protein.SHS27k.sequences.dictionary.pro3.tsv", help="File where to read protein sequences")
    parser.add_argument("--protein_seq_sep", default="\t", help=r"Separator between protein name and protein sequence. Default to '\t'")
    parser.add_argument("--all_for_assign", default="all_assign.txt", help="File where to read 'all_for_assign' variable")
    parser.add_argument("--out_adj", default="edge_list_12.npy", help="Output file for the adj list")
    parser.add_argument("--out_feat", default="x_list_7.pt", help="Output file for the feature list")
    parser.add_argument("--pdb_dest", default="pdbs", help="Destination folder for PDB files")

    return parser


def read_proteins(f_name, sep):
    pdbs = []
    with open(f_name) as f:
        for liness1 in tqdm(f):
            parts = liness1.split(sep)
            assert len(parts) == 2
            protein = parts[0]
            pdbs.append(protein)

    return pdbs


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


def collect_pdbs(pdbs, dest_folder):
    # Parallel download.
    print("  [DEBUG] Retrieving {} PDB files".format(len(pdbs)))
    errors = []
    with ThreadPool() as pool:
        result = pool.imap_unordered(worker_fn, zip(pdbs, cycle([dest_folder])))
        for f_name, status in tqdm(result, total=len(pdbs)):
            if not status:
                errors.append(f_name)
    # Dump errors
    if len(errors) > 0:
        print(f"  [DEBUG]: found {len(errors)} errors. "
              f"Dumping into {os.path.join(dest_folder, 'errors.txt')}")
        with open(os.path.join(dest_folder, "errors.txt"), "w") as f:
            for err in errors:
                f.write(f"{err}\n")

    return errors


def _compute_worker(all_for_assign, pdb_dir, distance, pdb_name):
    """Helper function to compute adjancey matrix and feature map in parallel."""
    pdb_file_name = os.path.join(pdb_dir, pdb_name + ".pdb")
    with open(pdb_file_name, "r") as f:
        contacts, xx = pdb_to_cm_and_ajs(f, distance)

    # Prepare the feature.
    x_p = np.zeros((len(xx), all_for_assign.shape[1]))
    for j in range(len(xx)):
        idx = PROTEIN_2_IDX_TABLE.get(xx[j])
        if idx is not None:
            x_p[j] = all_for_assign[idx, :]

    return contacts, x_p


def compute_adj_and_feature_lists(pdbs, distance, f_assign, pdb_dir):
    """Compute adjency matrix and feature map for each PDB in input."""
    # Read all the pdb files and extract interactions.
    print("  [DEBUG] Computing concat maps and adjacency lists.")

    all_for_assign = np.loadtxt(f_assign)
    assert len(all_for_assign.shape) == 2

    adj_list     = []
    feature_list = []

    f = partial(_compute_worker, all_for_assign, pdb_dir, distance)

    with Pool() as pool:
        result = pool.imap(f, pdbs, chunksize=64)
        for (contacts, x_p) in tqdm(result, total=len(pdbs)):
            adj_list.append(contacts)
            feature_list.append(x_p)

    return adj_list, feature_list


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    pdbs = read_proteins(args.protein_seq_file, args.protein_seq_sep)

    errors = collect_pdbs(pdbs, args.pdb_dest)

    errors = set(errors)
    if len(errors) > 0:  # filter out invalid pdbs
        pdbs = list(filter(lambda x: x not in errors, pdbs))

    adj_list, feature_list = compute_adj_and_feature_lists(
        pdbs, args.distance, args.all_for_assign, args.pdb_dest
    )

    # Save.
    adj_list = np.array(adj_list)
    np.save(args.out_adj, adj_list)
    torch.save(feature_list, args.out_feat)


if __name__ == "__main__":
    main()
