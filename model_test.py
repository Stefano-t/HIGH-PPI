import math
import numpy as np
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

from tqdm import tqdm
from gnn_models_sag import ppi_model
from gnn_data import GNN_DATA
from utils import Metrictor_PPI, multi2big_x, multi2big_batch, multi2big_edge


parser = argparse.ArgumentParser(description='HIGH-PPI_model_training')

parser.add_argument('--ppi_path', default=None, type=str,
                    help="ppi path")
parser.add_argument('--pseq_path', default=None, type=str,
                    help="protein sequence path")
parser.add_argument('--vec_path', default='./protein_info/vec5_CTC.txt', type=str,
                    help='protein sequence vector path')
parser.add_argument('--p_feat_matrix', default=None, type=str,
                    help="protein feature matrix")
parser.add_argument('--p_adj_matrix', default=None, type=str,
                    help="protein adjacency matrix")
parser.add_argument('--index_path', default=None, type=str,
                    help='training and test PPI index')
parser.add_argument('--model_path', default=None, type=str,
                    help="path for trained model")
parser.add_argument("--result_folder", default="results", type=str, help="Folder where to store results")


def test(model, graph, test_mask, device, batch, p_x_all, p_edge_all, result_file):
    model.eval()

    batch_size = 64   # @NOTE: same from paper!!!

    valid_steps = math.ceil(len(test_mask) / batch_size)

    valid_pre_result_list = []
    valid_label_list      = []
    true_prob_list        = []
    for step in tqdm(range(valid_steps)):
        if step == valid_steps-1:
            valid_edge_id = test_mask[step*batch_size:]
        else:
            valid_edge_id = test_mask[step*batch_size : step*batch_size + batch_size]

        output = model(batch, p_x_all, p_edge_all, graph.edge_index, valid_edge_id)
        label = graph.edge_attr_1[valid_edge_id]
        label = label.type(torch.FloatTensor).to(device)

        m = nn.Sigmoid()
        pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

        valid_pre_result_list.append(pre_result.cpu().data)
        valid_label_list.append(label.cpu().data)
        true_prob_list.append(m(output).cpu().data)

    valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
    valid_label_list      = torch.cat(valid_label_list, dim=0)
    true_prob_list        = torch.cat(true_prob_list, dim = 0)

    metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list, true_prob_list)

    metrics.show_result()

    print('recall: {}, precision: {}, F1: {}, AUPRC: {}'.format(metrics.Recall, metrics.Precision, \
        metrics.F1, metrics.Aupr))
    print(valid_pre_result_list)
    print(valid_label_list)

    # Dump to file
    with open(result_file, "w") as out:
        out.write("recall,precision,f1,auprc\n")
        for metric in [metrics.Recall, metrics.Precision, metrics.F1]:
            out.write(f"{metric},")
        out.write(f"{metrics.Aupr}\n")


def main():
    args = parser.parse_args()

    ppi_data = GNN_DATA(ppi_path=args.ppi_path)
    ppi_data.get_feature_origin(pseq_path=args.pseq_path,
                                vec_path=args.vec_path)


    ppi_data.generate_data()
    ppi_data.split_dataset(
        train_valid_index_path="./train_val_split_data/test.json",
        random_new=True,
        mode="test",
    )

    graph = ppi_data.data
    temp = graph.edge_index.transpose(0, 1).numpy()
    ppi_list = []

    for edge in temp:
        ppi_list.append(list(edge))

    graph.test_mask = ppi_data.ppi_split_dict["test_index"]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    graph.to(device)

    p_x_all    = torch.load(args.p_feat_matrix)
    p_edge_all = np.load(args.p_adj_matrix, allow_pickle=True)

    p_x_all, x_num_index = multi2big_x(p_x_all)
    p_edge_all, _ = multi2big_edge(p_edge_all, x_num_index)
    p_edge_all = p_edge_all - 1  # @NOTE: remove off-by-one error.

    batch = multi2big_batch(x_num_index) + 1

    model = ppi_model(
        class_num=p_x_all.shape[1],
        bgnn_hidden_size=128,
        tgnn_hidden_size=512,
    )
    model.to(device)
    model_path = args.model_path
    model.load_state_dict(torch.load(model_path)['state_dict'])

    result_folder = Path(args.result_folder)
    if not result_folder.is_dir():
        result_folder.mkdir()
    result_file = result_folder / datetime.now().isoformat()

    test(model, graph, graph.test_mask, device, batch, p_x_all, p_edge_all, result_file)

if __name__ == "__main__":
    main()
