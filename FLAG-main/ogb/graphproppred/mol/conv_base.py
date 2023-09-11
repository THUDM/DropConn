import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.utils import degree

import math
import pdb
import numpy as np


### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(
            x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, edge_hidden_channels, drop_ratio=0.5, JK="last", residual=False,
                 gnn_type='gin', adj_learn=1):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)
        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        self.edge_linear = torch.nn.Linear(emb_dim, 1)

    def forward(self, order, batched_data, perturb=None):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, \
                                          batched_data.batch
        new_fea_list = []
        for i in range(batched_data.y.shape[0]):
            new_fea = self.atom_encoder(batched_data[i].x)
            edge_index_fea = batched_data[i].edge_index
            N = new_fea.size(0)
            drop_rate = 0.2
            if self.training:
                drop_rates = torch.FloatTensor(np.ones(N) * drop_rate)
                masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
                new_fea = masks.to(new_fea.device) * new_fea
            else:
                new_fea = new_fea * (1. - drop_rate)
            edge_attr_in_batch = batched_data[i].edge_attr
            edge_attr_in_batch = self.bond_encoder(edge_attr_in_batch)
            edge_weight_emb = self.edge_linear(edge_attr_in_batch.float()).view(-1)
            edge_weight_emb = torch.sigmoid(edge_weight_emb)
            adj = get_adj_matrix(edge_index_fea, N, edge_weight_emb).to(edge_index.device)
            # order = 1
            new_fea = propagate(new_fea, adj, order)
            new_fea_list.append(new_fea)
        tmp = torch.cat(new_fea_list, dim=0) + perturb \
            if perturb is not None else torch.cat(new_fea_list, dim=0)
        h_list = [tmp]

        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]

        return node_representation


def get_adj_matrix(edge_index_fea, N):
    adj = torch.zeros([N, N])
    adj[edge_index_fea[0, :], edge_index_fea[1, :]] = 1
    adj[edge_index_fea[1, :], edge_index_fea[0, :]] = 1
    adj = adj + torch.eye(N)
    # print("adj: ", adj)
    # symmetric D^(-1/2)AD^(-1/2)
    degree_col = adj.sum(1)
    degree_row = adj.sum(0)
    r_col_inv = torch.pow(degree_col, -0.5)
    r_row_inv = torch.pow(degree_row, -0.5)
    r_mat_col_inv = torch.diag(r_col_inv)
    r_mat_row_inv = torch.diag(r_row_inv)
    A_ = r_mat_row_inv.mm(adj.mm(r_mat_col_inv))
    return A_


def get_adj_matrix_with_learn(edge_index_fea, N, edge_weight_emb):
    # learnable
    adj = torch.zeros([N, N], dtype=torch.float32).to(edge_index_fea.device)
    adj[edge_index_fea[0, :], edge_index_fea[1, :]] = edge_weight_emb
    adj = adj + torch.eye(N).to(edge_index_fea.device)
    degree_col = adj.sum(1)
    degree_row = adj.sum(0)
    r_col_inv = torch.pow(degree_col, -0.5)
    r_row_inv = torch.pow(degree_row, -0.5)
    r_mat_col_inv = torch.diag(r_col_inv)
    r_mat_row_inv = torch.diag(r_row_inv)
    A_ = r_mat_row_inv.mm(adj.mm(r_mat_col_inv))
    return A_


def propagate(feature, A, order):
    x = feature
    y = feature
    for i in range(order):
        x = torch.spmm(A, x).detach_()
        y.add_(x)
    return y.div_(order + 1.0).detach_()


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, edge_hidden_channels, drop_ratio=0.5, JK="last", residual=False,
                 gnn_type='gin', adj_learn=1):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(edge_hidden_channels)
        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        self.adj_learn = adj_learn
        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                    # torch.nn.BatchNorm1d(2 * emb_dim),
                                    torch.nn.ReLU(), \
                                    torch.nn.Linear(2 * emb_dim, emb_dim),
                                    # torch.nn.BatchNorm1d(emb_dim),
                                    torch.nn.ReLU()))

        self.edge_linear = torch.nn.Linear(edge_hidden_channels, 1)

    def forward(self, order, batched_data, perturb=None):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, \
                                          batched_data.batch

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        new_fea_list = []
        for i in range(batched_data.y.shape[0]):
            new_fea = self.atom_encoder(batched_data[i].x)
            edge_index_fea = batched_data[i].edge_index
            N = new_fea.size(0)
            if self.adj_learn == 1:
                edge_attr_in_batch = batched_data[i].edge_attr
                edge_attr_in_batch = self.bond_encoder(edge_attr_in_batch)
                edge_weight_emb = self.edge_linear(edge_attr_in_batch.float()).view(-1)
                edge_weight_emb = torch.sigmoid(edge_weight_emb)
                adj = get_adj_matrix_with_learn(edge_index_fea, N, edge_weight_emb).to(edge_index.device)
            else:
                adj = get_adj_matrix(edge_index_fea, N).to(edge_index.device)

            new_fea = propagate(new_fea, adj, order)
            new_fea_list.append(new_fea)
        tmp = torch.cat(new_fea_list, dim=0) + perturb \
            if perturb is not None else torch.cat(new_fea_list, dim=0)

        h_list = [tmp]

        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio,
                        training=self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                                                      self.drop_ratio, training=self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]

        return node_representation


if __name__ == "__main__":
    pass
