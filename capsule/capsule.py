from src.graph.rout import Dynamic_Routing
import torch.nn.functional as F
import torch.nn as nn
import torch

from torch.nn import init

# 21.06.2021 experiment
from diffpool.diffpool_v2 import SAGEConvolutions

from math import ceil

from torch_geometric.utils import to_dense_batch, to_dense_adj

class Capsule_Network(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations, dim_upsample, dim_features, mlp_hidden_dim,
                 max_num_nodes, num_layers, num_classes, capsule_mode):
        super().__init__()
        # run in dummy mode or capsule_mode
        self.capsule_mode = capsule_mode

        # this fully connected layer may be replaced with graph convolutional layer

        # 21.06.2021 experiment
        # self.linear_1 = nn.Linear(dim_features, dim_features)

        # 改hidden_channels out_channels to make it runnable
        self.gcn_emb = SAGEConvolutions(dim_features, dim_features, dim_features, lin=False)
        self.linear_2 = nn.Linear(dim_upsample, mlp_hidden_dim)
        self.linear_3 = nn.Linear(mlp_hidden_dim, num_classes)

        # specify parameters for capsule module
        self.input_caps = input_caps
        self.output_caps = output_caps
        self.max_num_nodes = max_num_nodes
        self.input_dim = input_caps
        # in most general case: hidden + hidden + out (SAGEConvolutions)?
        self.dim_hidden = 3 * dim_features  # dimension of hidden node features after 1 gcn layer

        # a dummy replacement for capsule module
        # TODO: find a variable that can represent 111 !!! Done!
        self.linear_dummy = nn.Linear(self.dim_hidden, dim_upsample)

        # just for the if statement in train.py: if linkpred and model.pooling_type == 'gnn':
        self.pooling_type = None

        # Reproduce paper choice about coarse factor
        coarse_factor = 0.1 if num_layers == 1 else 0.25
        output_caps = ceil(input_caps * coarse_factor)
        # TODO: find a variable that can represent 111 !!! Done!
        self.dynamic_routing = Dynamic_Routing(input_caps, self.dim_hidden, output_caps, dim_upsample, n_iterations)

    # def feature_nodes_extraction_with_masking(self, data, unique_graph_idxs):
    #     features_graphs = []
    #     maskings = []
    #     for idx in unique_graph_idxs:
    #         features_graph = data.x[data.batch == idx]
    #         num_padding = self.max_num_nodes - features_graph.shape[0]
    #
    #         masking = torch.ones(len(features_graph)).cuda()
    #         masking = F.pad(input=masking, pad=(0, num_padding),
    #                                mode='constant', value=0).unsqueeze(0)
    #
    #         features_graph = F.pad(input=features_graph, pad=(0, 0, 0, num_padding),
    #                                mode='constant', value=0).unsqueeze(0)
    #         maskings.append(masking)
    #         features_graphs.append(features_graph)
    #
    #     return torch.cat(features_graphs, dim=0), torch.cat(maskings, dim=0)

    def feature_nodes_extraction_with_masking_GCNmodule(self, data, unique_graph_idxs):
        # padding adjacency matrix
        adj = to_dense_adj(data.edge_index, batch=data.batch)
        num_padding_adj = self.max_num_nodes - adj.shape[1]
        adj = F.pad(input=adj, pad=(0, num_padding_adj, 0, num_padding_adj, 0, 0),
                    mode='constant', value=0)

        features_graphs = []
        maskings = []

        # padding masking and features_graph
        for idx in unique_graph_idxs:
            features_graph = data.x[data.batch == idx]
            num_padding = self.max_num_nodes - features_graph.shape[0]

            masking = torch.ones(len(features_graph)).cuda()
            masking = F.pad(input=masking, pad=(0, num_padding),
                                   mode='constant', value=0).unsqueeze(0)

            features_graph = F.pad(input=features_graph, pad=(0, 0, 0, num_padding),
                                   mode='constant', value=0).unsqueeze(0)
            maskings.append(masking)
            features_graphs.append(features_graph)

        return torch.cat(features_graphs, dim=0), torch.cat(maskings, dim=0), adj


    def forward(self, data, prob=False):
        unique_graph_idxs = data.batch.unique()

        feature_nodes, masking, adj = self.feature_nodes_extraction_with_masking_GCNmodule(data, unique_graph_idxs)

        feature_nodes = self.gcn_emb(feature_nodes, adj, masking)

        # TODO: make multiple capsule layers (comparable to diffpool model)

        if self.capsule_mode == 'capsule':
            v = F.relu(self.dynamic_routing(feature_nodes, prob))  # if use capsule mode
        elif self.capsule_mode == 'dummy':
            v = F.relu(self.linear_dummy(feature_nodes))  # if use dummy mode

        v = torch.max(v, dim=1)[0]
        v = F.relu(self.linear_2(v))
        v = self.linear_3(v)


        # 0 works as placeholder for l_total and e_total?
        return v, torch.tensor(0.).cuda(), torch.tensor(0.).cuda()

def init_parameters(capsule_network):
    for m in capsule_network.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                init.constant_(m.bias.data, 0.0)

