import numpy as np
import random
from sklearn.metrics import precision_recall_curve, auc
import torch


# *** Training / Test utils ***

def multi2big_x(x_ori):
    """Transforms the input list of arrays into a single array. Second dim must match between all arrays.

    Transfroms like (L, M, N) -> (L*M, N).
    """
    input_len = len(x_ori)
    input_second_dim = x_ori[0].shape[1]
    for elt in x_ori:
        assert elt.shape[1] == input_second_dim, "Second dim doesn't match"

    x_cat = torch.zeros(1, input_second_dim)
    x_num_index = torch.zeros(input_len)
    for i in range(input_len):
        x_now = torch.tensor(x_ori[i])
        x_num_index[i] = torch.tensor(x_now.size(0))
        x_cat = torch.cat((x_cat, x_now), 0)
    return x_cat[1:, :], x_num_index


def multi2big_batch(x_num_index):
    """Assigns an index to each consecutive index.

    That is, index at position `x_num_index[i]` will generate a batch `x_num_index[i+1] - x_num_index[i]` of value `i`.
    """
    num_sum = x_num_index.sum()
    num_sum = num_sum.int()
    batch = torch.zeros(num_sum)
    count = 1
    cumsum = x_num_index.cumsum(0, dtype=torch.int)
    for i in range(1, len(x_num_index)):
        zj11 = cumsum[i-1]
        zj22 = zj11 + x_num_index[i]
        zj22 = zj22.int()
        size1 = x_num_index[i]
        size1 = size1.int()
        tc = count * torch.ones(size1)
        batch[zj11:zj22] = tc
        count = count + 1
    batch = batch.int()
    return batch


def multi2big_edge(edge_ori, num_index):
    """Transfroms the input list of edges into a single edge array.

    Applies the transformation: (L, E, N) -> (N, L * E).
    """
    assert len(edge_ori) > 0
    assert len(edge_ori[0]) > 0

    edge_len = len(edge_ori[0][0])
    edge_cat = torch.zeros(edge_len, 1)

    offsets = num_index.cumsum(0)
    input_len = len(edge_ori)
    edge_num_index = torch.zeros(input_len)
    for i in range(input_len):
        edge_index_p = edge_ori[i]
        edge_index_p = np.asarray(edge_index_p)
        edge_index_p = torch.tensor(edge_index_p.T)
        edge_num_index[i] = torch.tensor(edge_index_p.size(1))
        if i == 0:
            offset = 0
        else:
            offset = offsets[i-1]
        edge_cat = torch.cat((edge_cat, edge_index_p + offset), 1)
    return edge_cat[:, 1:], edge_num_index


# *** Misc utils ***

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def print_file(str_, save_file_path=None):
    print(str_)
    if save_file_path is not None:
        f = open(save_file_path, 'a')
        print(str_, file=f)


class Metrictor_PPI:
    def __init__(self, pre_y, truth_y, true_prob, is_binary=False):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.pre = np.array(pre_y).squeeze()
        self.tru = np.array(truth_y).squeeze()
        self.true_prob = np.array(true_prob).squeeze()
        if is_binary:
            length = pre_y.shape[0]
            for i in range(length):
                if pre_y[i] == truth_y[i]:
                    if truth_y[i] == 1:
                        self.TP += 1
                    else:
                        self.TN += 1
                elif truth_y[i] == 1:
                    self.FN += 1
                elif pre_y[i] == 1:
                    self.FP += 1
            self.num = length

        else:
            N, C = pre_y.shape
            for i in range(N):
                for j in range(C):
                    if pre_y[i][j] == truth_y[i][j]:
                        if truth_y[i][j] == 1:
                            self.TP += 1
                        else:
                            self.TN += 1
                    elif truth_y[i][j] == 1:
                        self.FN += 1
                    elif truth_y[i][j] == 0:
                        self.FP += 1
            self.num = N * C

    def show_result(self, is_print=False, file=None):
        self.Accuracy = (self.TP + self.TN) / (self.num + 1e-10)
        self.Precision = self.TP / (self.TP + self.FP + 1e-10)
        self.Recall = self.TP / (self.TP + self.FN + 1e-10)
        self.F1 = 2 * self.Precision * self.Recall / (self.Precision + self.Recall + 1e-10)
        aupr_entry_1 = self.tru
        aupr_entry_2 = self.true_prob
        aupr = np.zeros(7)
        for i in range(7):
            precision, recall, _ = precision_recall_curve(aupr_entry_1[:,i], aupr_entry_2[:,i])
            aupr[i] = auc(recall,precision)
        self.Aupr = aupr

        if is_print:
            print_file("Accuracy: {}".format(self.Accuracy), file)
            print_file("Precision: {}".format(self.Precision), file)
            print_file("Recall: {}".format(self.Recall), file)
            print_file("F1-Score: {}".format(self.F1), file)


class UnionFindSet(object):
    def __init__(self, m):
        self.roots = [i for i in range(m)]
        self.rank = [0 for i in range(m)]
        self.count = m

        for i in range(m):
            self.roots[i] = i

    def find(self, member):
        tmp = []
        while member != self.roots[member]:
            tmp.append(member)
            member = self.roots[member]
        for root in tmp:
            self.roots[root] = member
        return member

    def union(self, p, q):
        parentP = self.find(p)
        parentQ = self.find(q)
        if parentP != parentQ:
            if self.rank[parentP] > self.rank[parentQ]:
                self.roots[parentQ] = parentP
            elif self.rank[parentP] < self.rank[parentQ]:
                self.roots[parentP] = parentQ
            else:
                self.roots[parentQ] = parentP
                self.rank[parentP] -= 1
            self.count -= 1


def get_bfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    candiate_node = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 5:
        random_node = random.randint(0, node_num - 1)
    candiate_node.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        cur_node = candiate_node.pop(0)
        selected_node.append(cur_node)
        for edge_index in node_to_edge_index[cur_node]:

            if edge_index not in selected_edge_index:
                selected_edge_index.append(edge_index)

                end_node = -1
                if ppi_list[edge_index][0] == cur_node:
                    end_node = ppi_list[edge_index][1]
                else:
                    end_node = ppi_list[edge_index][0]

                if end_node not in selected_node and end_node not in candiate_node:
                    candiate_node.append(end_node)
            else:
                continue

    return selected_edge_index


def get_dfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    stack = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 5:
        random_node = random.randint(0, node_num - 1)
    stack.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        cur_node = stack[-1]
        if cur_node in selected_node:
            flag = True
            for edge_index in node_to_edge_index[cur_node]:
                if flag:
                    end_node = -1
                    if ppi_list[edge_index][0] == cur_node:
                        end_node = ppi_list[edge_index][1]
                    else:
                        end_node = ppi_list[edge_index][0]

                    if end_node in selected_node:
                        continue
                    else:
                        stack.append(end_node)
                        flag = False
                else:
                    break
            if flag:
                stack.pop()
            continue
        else:
            selected_node.append(cur_node)
            for edge_index in node_to_edge_index[cur_node]:
                if edge_index not in selected_edge_index:
                    selected_edge_index.append(edge_index)

    return selected_edge_index
