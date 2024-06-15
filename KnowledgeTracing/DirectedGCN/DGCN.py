import torch
import torch.nn as nn
import torch.nn.functional as F

class DGCNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(DGCNNLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_features, out_features, kernel_size=1)
        self.conv2 = nn.Conv1d(out_features, out_features, kernel_size=1)

    def forward(self, x, adj):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = torch.bmm(adj, x)
        return x


class DGCNN(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super(DGCNN, self).__init__()
        self.dgcnn_layer1 = DGCNNLayer(in_features, hidden_size)
        self.dgcnn_layer2 = DGCNNLayer(hidden_size, out_features)

    def forward(self, x, adj):
        x = self.dgcnn_layer1(x, adj)
        x = F.relu(x)
        x = self.dgcnn_layer2(x, adj)
        return F.log_softmax(x, dim=1)