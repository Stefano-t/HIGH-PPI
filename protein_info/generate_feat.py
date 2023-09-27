import numpy as np
from tqdm import tqdm
import torch
from generate_utils import pdb_to_x, PROTEIN_2_IDX_TABLE

c = 0
count1 = -1
ensp = open('ensp_uniprot.txt')
e = ensp.read()
e_sp = e.split('ENSP')
list_all = []
x_set = np.zeros(())
all_for_assign = np.loadtxt("all_assign.txt")
for liness1 in tqdm(open('protein.SHS27k.sequences.dictionary.pro3.tsv')):
    count1 = count1 + 1
    line1 = liness1.split('\t')
    li = line1[0][10:]
    for i in range(1690):
        e_zj = e_sp[i]
        res = li in e_zj
        if res:
            li2 = e_zj[13:-9]
            pdb_file_name = li2 + '.pdb'
            print(pdb_file_name)
            c = c + 1
            xx = pdb_to_x(open(pdb_file_name, "r"), 7.5)
            break

    x_p = np.zeros((len(xx), 7))
    for j in range(len(xx)):
        idx = PROTEIN_2_IDX_TABLE.get(xx[j])
        if idx is not None:
            x_p[j] = all_for_assign[idx, :]

    list_all.append(x_p)

torch.save(list_all,'x_list_7.pt')
