import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
# from pygcn.layers import GraphConvolution
# from dgl.nn import GraphConv, EdgeWeightNorm
from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool, SAGEConv, GCNConv
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import global_mean_pool



class GIN(torch.nn.Module):
    def __init__(self,  hidden=512, train_eps=True, class_num=7):
        super(GIN, self).__init__()
        self.train_eps = train_eps
        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(128, hidden),  # @NOTE: 128 is the output dim of BGNN (or GCN)
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                # nn.Linear(hidden, hidden),
                # nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                # nn.Linear(hidden, hidden),
                # nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv3 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )

        self.lin1 = nn.Linear(hidden, hidden)
        self.fc1 = nn.Linear(2 * hidden, class_num) #clasifier for concat
        self.fc2 = nn.Linear(hidden, class_num)   #classifier for inner product



    def reset_parameters(self):

        self.fc1.reset_parameters()

        self.gin_conv1.reset_parameters()
        self.gin_conv2.reset_parameters()
        # self.gin_conv3.reset_parameters()
        self.lin1.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


    def forward(self, x, edge_index, train_edge_id, p=0.5):

        x = self.gin_conv1(x, edge_index)
        x = self.gin_conv2(x, edge_index)
        # x = self.gin_conv3(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        node_id = edge_index[:, train_edge_id]
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]
        # x = torch.cat([x1, x2], dim=1)
        # x = self.fc1(x)
        x = torch.mul(x1, x2)
        x = self.fc2(x)


        return x



class GCN(nn.Module):
    def __init__(self, class_num: int = 7, hidden_size: int = 128):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(class_num, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.conv4 = GCNConv(hidden_size, hidden_size)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)

        self.sag1 = SAGPooling(hidden_size,0.5)
        self.sag2 = SAGPooling(hidden_size,0.5)
        self.sag3 = SAGPooling(hidden_size,0.5)
        self.sag4 = SAGPooling(hidden_size,0.5)

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(0.5)
        for param in self.parameters():
            print(type(param), param.size())


    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        y = self.sag1(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv2(x, edge_index)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)
        y = self.sag2(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv3(x, edge_index)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.bn3(x)
        y = self.sag3(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv4(x, edge_index)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.bn4(x)
        y = self.sag4(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        # y = self.sag4(x, edge_index, batch = batch)

        return global_mean_pool(y[0], y[3])
        # return y[0]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class ppi_model(nn.Module):
    def __init__(self):
        super(ppi_model,self).__init__()
        self.BGNN = GCN()
        self.TGNN = GIN()

    def forward(self, batch, p_x_all, p_edge_all, edge_index, train_edge_id, p=0.5):
        edge_index = edge_index.to(device)
        batch = batch.to(torch.int64).to(device)
        x = p_x_all.to(torch.float32).to(device)
        edge = torch.LongTensor(p_edge_all.to(torch.int64)).to(device)
        embs = self.BGNN(x, edge, batch-1)
        final = self.TGNN(embs, edge_index, train_edge_id, p=0.5)
        return final
