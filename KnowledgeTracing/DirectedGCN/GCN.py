import torch.nn as nn
import torch.nn.functional as F
from KnowledgeTracing.DirectedGCN.layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)


    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x2 = F.relu(self.gc2(x1, adj))

        return x2



'''
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

    def forward(self, x, adj, prev_output=None):
        x1 = F.relu(self.gc1(x, adj))
        if prev_output is not None:
            x1 += prev_output  # 添加跨层连接
        x2 = F.relu(self.gc2(x1, adj))

        return x2
'''
#跨层连接机制：引入跨层连接机制，使得网络能够更好地捕捉多层次的结构信息。这种机制有助于提高模型对图中复杂关系的感知能力，从而提高图神经网络在有向图上的表现能力。