from __future__ import print_function

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# utils functions
def squash(x):
    lengths2 = x.pow(2).sum(dim=2)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x


# Dynamic Routing Agreement
# class Dynamic_Routing(nn.Module):
#     def __init__(self, input_caps, output_caps, n_iterations):
#         super(Dynamic_Routing, self).__init__()
#         self.n_iterations = n_iterations
#         self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))
#
#     def forward(self, u_predict, prob=True):
#         batch_size, input_caps, output_caps, output_dim = u_predict.size()
#
#         c = F.softmax(self.b, dim=-1)
#         s = (c.unsqueeze(2) * u_predict).sum(dim=1)
#         v = squash(s)
#
#         if self.n_iterations > 0:
#             b_batch = self.b.expand((batch_size, input_caps, output_caps))
#
#             for r in range(self.n_iterations):
#                 v = v.unsqueeze(1)
#                 b_batch = b_batch + (u_predict * v).sum(-1)
#
#                 c = F.softmax(b_batch.view(-1, output_caps), dim=-1).view(-1, input_caps, output_caps, 1)
#                 s = (c * u_predict).sum(dim=1)
#                 v = squash(s)
#
#         if prob:
#             # output: probability for each class, will be fed to loss function
#             return v.pow(2).sum(dim=2).sqrt()
#         else:
#             return v


# my assumption
# input_caps: maximum number of nodes (num of input capsules)
# input_dim: dimension of node features (dimension of each input capsule)
# output_caps: num of classes (num of output capsules)
# output_dim: dimension of each output capsule (dimension should be shrinked to 1 in the end)
# Dynamic Routing Agreement
class Dynamic_Routing(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, n_iterations):
        super(Dynamic_Routing, self).__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))

        self.input_caps = input_caps
        self.output_caps = output_caps
        self.output_dim = output_dim
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_input, prob=True):
        caps_input = caps_input.unsqueeze(2)

        u_predict = caps_input.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)

        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b, dim=-1)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))

            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = F.softmax(b_batch.view(-1, output_caps), dim=-1).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)

        if prob:
            # return probability of each class
            return v.pow(2).sum(dim=2).sqrt()
        else:
            return v # batchsize * num_super_nodes * upsampling_dim
    
    

class Self_Attention_Routing(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, u_predict):
        # return a list of vectors
        pass

# a short demo
if __name__ == '__main__':
    # one_graph = 50 * torch.ones([1, 50, 30])
    Batch_Size = 4
    Input_Caps = 50 # depends on the maximum number of nodes
    Output_Caps = 11 # depends on the number of class
    N_Iterations = 2
    N_Features = 30 # depends on dimension of node feature
    Output_Features = 160 # depends on dim_upsample

    graphs = 1000 * torch.randn([Batch_Size, Input_Caps, N_Features])
    masking = torch.ones([Batch_Size, 25])
    masking = torch.cat((masking, torch.zeros([Batch_Size, Input_Caps-25])), dim=1).unsqueeze(-1)
    #masking = torch.zeros([1, 50])
    one_graph = graphs * masking

    dynamic_routing = Dynamic_Routing(input_caps=Input_Caps, input_dim=N_Features, output_caps=Output_Caps,
                                      output_dim=Output_Features, n_iterations=N_Iterations)
    v = dynamic_routing(one_graph, prob=False)