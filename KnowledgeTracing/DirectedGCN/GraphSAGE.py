import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGEConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GraphSAGEConv, self).__init__()
        self.linear = nn.Linear(in_feats * 2, out_feats)

    def forward(self, h, adj_matrix):
        neigh_agg = torch.matmul(adj_matrix, h)  # 根据邻接矩阵聚合邻居节点特征
        combined = torch.cat([h, neigh_agg], dim=1)
        return F.relu(self.linear(combined))

class GraphSAGEModel(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, adj_matrix):
        super(GraphSAGEModel, self).__init__()
        self.adj_matrix = adj_matrix
        self.conv1 = GraphSAGEConv(in_feats, hidden_size)
        self.conv2 = GraphSAGEConv(hidden_size, out_feats)

    def forward(self, features):
        h = self.conv1(features, self.adj_matrix)
        h = self.conv2(h, self.adj_matrix)
        return h
