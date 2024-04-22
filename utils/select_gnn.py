import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '../nets'))
from rgcnDGL import RGCN
from gat import GAT
from mpnn_dgl import MPNN
from Tconv import ResnetGenerator
import dgl
import functools
import torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(__file__), '../dataset'))
from alternatives_utils import grid_width, image_width


def get_activation_function(activation_name):
    if activation_name == 'elu':
        activation = torch.nn.ELU()
    elif activation_name == 'leaky_relu':
        activation = torch.nn.LeakyReLU()
    elif activation_name == 'relu':
        activation = torch.nn.ReLU()
    elif activation_name == 'tanh':
        activation = torch.nn.Tanh()
    elif activation_name == 'sigmoid':
        activation = torch.nn.Sigmoid()
    else:
        activation = None

    return activation


class SELECT_GNN(nn.Module):
    def __init__(self, num_features, num_edge_feats, n_classes, num_hidden, gnn_layers, cnn_layers, dropout,
                 activation, final_activation, activation_cnn, final_activation_cnn, num_channels, gnn_type, num_heads,
                 num_rels, num_bases, g, residual, aggregator_type, attn_drop, K=0, num_hidden_layers_rgcn=0,
                 num_hidden_layers_gat=0, num_hidden_layer_pairs=0, improved=True, concat=True, neg_slope=0.2,
                 bias=True, norm=None, alpha=0.12):
        super(SELECT_GNN, self).__init__()

        self.activation = get_activation_function(activation)
        self.final_activation = get_activation_function(final_activation)
        self.activation_cnn = get_activation_function(activation_cnn)
        self.final_activation_cnn = get_activation_function(final_activation_cnn)
        self.num_hidden_layer_pairs = num_hidden_layer_pairs
        self.attn_drop = attn_drop
        self.num_hidden_layers_rgcn = num_hidden_layers_rgcn
        self.num_hidden_layers_gat = num_hidden_layers_gat
        self.num_rels = num_rels
        self.residual = residual
        self.aggregator = aggregator_type
        self.num_bases = num_bases
        self.num_channels = num_channels
        self.n_classes = n_classes
        self.num_hidden = num_hidden
        self.gnn_layers = gnn_layers
        self.cnn_layers = cnn_layers
        self.num_features = num_features
        self.num_edge_feats = num_edge_feats
        self.dropout = dropout
        self.bias = bias
        self.norm = norm
        self.improved = improved
        self.K = K
        self.g = g
        self.num_heads = num_heads
        self.concat = concat
        self.neg_slope = neg_slope
        self.dropout1 = dropout
        self.alpha = alpha
        self.gnn_type = gnn_type
        self.grid_width = grid_width
        self.image_width = image_width
        self.cnn_object = self.create_cnn()

        if self.dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Dropout(p=0.)
        #
        if self.gnn_type == 'rgcn':
            print("GNN being used is RGCN")
            self.gnn_object = self.rgcn()
        elif self.gnn_type == 'gat':
            print("GNN being used is GAT")
            self.gnn_object = self.gat()
        elif self.gnn_type == 'mpnn':
            print("GNN being used is MPNN")
            self.gnn_object = self.mpnn()

    def rgcn(self):
        return RGCN(self.g, self.gnn_layers, self.num_features, self.num_channels, self.num_hidden, self.num_rels, self.activation, self.final_activation, self.dropout1, self.num_bases)

    def gat(self):
        return GAT(self.g, self.gnn_layers, self.num_features, self.num_channels, self.num_hidden, self.num_heads, self.activation, self.final_activation,  self.dropout1, self.attn_drop, self.alpha, self.residual)

    def mpnn(self):
        return MPNN(self.num_features, self.num_channels, self.num_hidden, self.num_edge_feats, self.final_activation, self.aggregator, self.bias, self.residual, self.norm, self.activation)

    def create_cnn(self):
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        return ResnetGenerator(self.num_channels, output_nc=self.n_classes, ngf=64, norm_layer=norm_layer,
                               use_dropout=self.dropout, n_blocks=9)

    def forward(self, data, g, efeat):
        if self.gnn_type in ['gatmc', 'prgat2', 'prgat3']:
            x = self.gnn_object(data)
        elif self.gnn_type == 'mpnn':
            x = self.gnn_object(g, data, efeat)
        else:
            x = self.gnn_object(data, g)

        logits = x.squeeze()
        
        base_index = 0
        batch_number = 0
        unbatched = dgl.unbatch(self.g)
        
        cnn_input = torch.Tensor(size=(len(unbatched), self.num_channels, self.grid_width, self.grid_width)).to(data.device)
        for g in unbatched:
            num_nodes = g.number_of_nodes()
            
            first = base_index
            last =  base_index + self.grid_width*self.grid_width - 1

            x_prev = logits[first:last+1, :].view(self.grid_width, self.grid_width, self.num_channels)
            
            x_permute = x_prev.permute(2, 0, 1).flatten(1)
            x = x_permute.view(self.num_channels, self.grid_width, self.grid_width)
            cnn_input[batch_number, :, :, :] = x
            
            base_index += num_nodes
            batch_number += 1

        cnn_input = F.interpolate(cnn_input, size=(256, 256), mode='nearest')
        output = self.cnn_object(cnn_input)

        return output
