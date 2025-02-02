import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, GCNConv
from torch_geometric.nn.pool import SAGPooling


# @NOTE: paper states that GIN blocks are 3, but only 2 are actually used during
# the `forward` phase. Furthemore, the GIN block itself contains 2 Linear and 2
# RELU, while the paper states only 1 of each layer is used...
class GIN(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden: int = 512,
        train_eps: bool = True,
        class_num: int = 7,
    ):
        super(GIN, self).__init__()
        self.train_eps = train_eps
        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(input_size, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
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
        self.lin1.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


    def forward(self, x, edge_index, train_edge_id):

        x = self.gin_conv1(x, edge_index)
        x = self.gin_conv2(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        node_id = edge_index[:, train_edge_id]
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]
        x = torch.mul(x1, x2)
        x = self.fc2(x)


        return x


# @NOTE: the paper states just 2 GCN layers... here we have 4!!!
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

        return global_mean_pool(y[0], y[3])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class ppi_model(nn.Module):
    def __init__(
            self, class_num: int, bgnn_hidden_size: int, tgnn_hidden_size: int
    ):
        super(ppi_model,self).__init__()
        self.BGNN = GCN(class_num=class_num, hidden_size=bgnn_hidden_size)
        self.TGNN = GIN(bgnn_hidden_size, hidden=tgnn_hidden_size, class_num=class_num)

    def forward(self, batch, p_x_all, p_edge_all, edge_index, train_edge_id):
        edge_index = edge_index.to(device)
        batch = batch.to(torch.int64).to(device)
        x = p_x_all.to(torch.float32).to(device)
        edge = torch.LongTensor(p_edge_all.to(torch.int64)).to(device)
        embs = self.BGNN(x, edge, batch-1)
        final = self.TGNN(embs, edge_index, train_edge_id)
        return final
