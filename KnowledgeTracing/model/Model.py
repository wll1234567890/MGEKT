# -*- coding: utf-8 -*-
# @Time : 2021/12/23 13:43
# @Author : Yumo
# @File : Model.py
# @Project: GOODKT
# @Comment :

import networkx as nx

from KnowledgeTracing.hgnn_models import HGNN
from KnowledgeTracing.Constant import Constants as C
import torch.nn as nn
import torch
from KnowledgeTracing.DirectedGCN.GCN import GCN
from KnowledgeTracing.DirectedGCN.AGCN import AGCN
from KnowledgeTracing.DirectedGCN.DGCN import DGCNN
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
from torchdiffeq import odeint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(DGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru_cells = nn.ModuleList()
        for i in range(num_layers):
            self.gru_cells.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

        self.gate_linear = nn.Linear(hidden_size * num_layers, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        hidden_list = []
        h_t = input
        for i in range(self.num_layers):
            h_t = self.gru_cells[i](h_t, hidden[i])
            hidden_list.append(h_t)

        hidden_cat = torch.cat(hidden_list, dim=1)
        gate = self.sigmoid(self.gate_linear(hidden_cat))
        h_t = gate * h_t + (1 - gate) * input

        return h_t


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if torch.is_tensor(sparse_mx):
        sparse_mx = sparse_mx.to_dense()
    sparse_mx = sp.coo_matrix(sparse_mx.cpu().numpy(), dtype=np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def normalize(mx, device):
    """Row-normalize sparse matrix."""
    rowsum = mx.sum(1).cpu().numpy()  # 将结果从cuda设备转移到主机内存
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = torch.tensor(r_mat_inv.dot(mx.cpu().numpy())).to(device)  # 将结果从cuda设备转移到主机内存
    return mx


class DKT(nn.Module):

    def __init__(self, hidden_dim, layer_dim, G, adj_in, adj_out):
        super(DKT, self).__init__()
        '''initial feature'''
        emb_dim = C.EMB
        emb = nn.Embedding(2 * C.NUM_OF_QUESTIONS, emb_dim)
        self.ques = emb(torch.LongTensor([i for i in range(2 * C.NUM_OF_QUESTIONS)])).cuda()
        emb1 =nn.Embedding(C.NUM_OF_QUESTIONS, emb_dim)
        self.ques1 = emb1(torch.LongTensor([i for i in range( C.NUM_OF_QUESTIONS)])).cuda()
        '''generate two graphs'''
        self.G = G
        self.adj_out = adj_out
        self.adj_in = adj_in
        '''DGCN'''
        self.net1 = GCN(nfeat=C.EMB, nhid=C.EMB, nclass=int(C.EMB / 2))
        self.net2 = GCN(nfeat=C.EMB, nhid=C.EMB, nclass=int(C.EMB / 2))

        #self.net1 = AGCN(nfeat=C.EMB, nhid=C.EMB, nclass=int(C.EMB / 2), dropout=0.8, alpha=0.2)
        #self.net2 = AGCN(nfeat=C.EMB, nhid=C.EMB, nclass=int(C.EMB / 2), dropout=0.8, alpha=0.2)
        adj_matrix1 = np.loadtxt('../../Dataset/kg_pk17xin.edgelist', delimiter=',')
        # 转换为张量
        self.adj_matrix = torch.tensor(adj_matrix1, dtype=torch.float32).to(device)


        '''HGCN'''
        self.net = HGNN(in_ch=C.EMB,
                        n_hid=C.EMB,
                        n_class=C.EMB)
        '''GRU'''
        self.rnn1 = nn.GRU(C.EMB, hidden_dim, layer_dim, batch_first=True)
        self.rnn2 = nn.GRU(C.EMB, hidden_dim, layer_dim, batch_first=True)
        #self.rnn1 = GRUODEBayes(C.EMB, hidden_dim)  # 注意：这里需要根据实际参数调整
        #self.rnn2 = GRUODEBayes(C.EMB, hidden_dim)

        self.attn1 = nn.MultiheadAttention(embed_dim=C.EMB, num_heads=8)  # 根据需求设置 num_heads 的值
        self.attn2 = nn.MultiheadAttention(embed_dim=C.EMB, num_heads=8)  # 根据需求设置 num_heads 的值
        '''kd'''
        self.fc_c = nn.Linear(hidden_dim, C.NUM_OF_QUESTIONS)
        self.fc_t = nn.Linear(hidden_dim, C.NUM_OF_QUESTIONS)
        self.fc_ensemble = nn.Linear(2 * hidden_dim, C.NUM_OF_QUESTIONS)
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)

        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(128, 256)

    def forward(self, x):
        '''SkillGraph: HGCN'''
        # ques_h = self.net(self.ques, self.G)
        '''TransitionGraph: DGCN'''

        ques_out = self.net1(self.ques, self.adj_out)
        ques_in = self.net2(self.ques, self.adj_in)
        ques_d = torch.cat([ques_in, ques_out], -1)
        '''choose 50'''

        loaded_array1 = np.load('../../Dataset/node2vec/emb/ASSIST17_kg_cw1_128_80_10_5_20_1.00_1.00.embQ.npy')
        tensor_array1 = torch.tensor(loaded_array1)
        tensor_array1 = tensor_array1.to(device)
        tensor_array_expanded1 = tensor_array1.repeat(2, 1)

        loaded_array2 = np.load('../../Dataset/node2vec/emb/ASSIST17_kg_pk17_128_80_10_5_20_1.00_1.00.embQ.npy')
        tensor_array2 = torch.tensor(loaded_array2)
        tensor_array2 = tensor_array2.to(device)
        tensor_array_expanded2 = tensor_array2.repeat(2, 1)

        loaded_array3 = np.load('../../Dataset/node2vec/emb/ASSIST17_kg_cw_128_80_10_5_20_1.00_1.00.embQ.npy')
        tensor_array3 = torch.tensor(loaded_array3)
        tensor_array3 = tensor_array3.to(device)
        tensor_array_expanded3 = tensor_array3.repeat(2, 1)

        loaded_array4 = np.load('../../Dataset/node2vec/emb/ASSIST17_kg_pk_128_80_10_5_20_1.00_1.00.embQ.npy')
        tensor_array4 = torch.tensor(loaded_array4)
        tensor_array4 = tensor_array4.to(device)
        tensor_array_expanded4 = tensor_array4.repeat(2, 1)

        embedding_tensors = [ tensor_array_expanded1, tensor_array_expanded2, tensor_array_expanded3 ,tensor_array_expanded4]

        #embedding_tensors = [tensor_array_expanded, tensor_array_expanded1]

        pooled_embedding = torch.stack(embedding_tensors, dim=0).mean(dim=0)


        # 执行加权求和
        ques_h = pooled_embedding.float()

        ques_h = self.linear(ques_h)
        x_h = x.matmul(ques_h)
        x_d = x.matmul(ques_d)

        '''gru'''
        out_h, _ = self.rnn1(x_h)
        out_d, _ = self.rnn2(x_d)


        '''logits'''
        logit_c = self.fc_c(out_h)
        logit_t = self.fc_t(out_d)

        '''kd'''
        '''
        theta = self.sigmoid(self.w1(out_h) + self.w2(out_d))
        out_d = theta * out_d
        out_h = (1 - theta) * out_h
        '''

        attention_scores = torch.matmul(out_h, out_d.transpose(1, 2))  # 计算注意力分数
        attention_weights = F.softmax(attention_scores, dim=2)  # 计算注意力权重
        # 使用注意力权重对两个状态进行加权
        out_d = torch.bmm(attention_weights, out_d)
        out_h = torch.matmul(1 - attention_weights, out_h)

        emseble_logit = self.fc_ensemble(torch.cat([out_d, out_h], -1))
        return logit_c, logit_t, emseble_logit

