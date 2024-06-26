from torch.utils.data import DataLoader
import dgl
import torch
import numpy as np
import sys
import os
import pickle
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'nets'))

from select_gnn import SELECT_GNN

global g_device


def collate(batch):

    graphs = [batch[0][0]]
    labels = batch[0][1]
    for graph, label in batch[1:]:
        graphs.append(graph)
        labels = torch.cat([labels, label], dim=0)
    batched_graphs = dgl.batch(graphs).to(torch.device(g_device))
    labels.to(torch.device(g_device))

    return batched_graphs, labels


def collate_list(batch):

    graphs = [batch[0].graphs[0]]
    labels = batch[0].labels[0]
    for g in batch[1:]:
        graphs.append(g.graphs[0])
        labels = torch.cat([labels, g.labels[0]], dim=0)
    batched_graphs = dgl.batch(graphs).to(torch.device(g_device))
    labels.to(torch.device(g_device))

    return batched_graphs, labels


class SocNavAPI(object):
    def __init__(self, base=None, dataset=None, device='cpu'):
        self.device = torch.device(device)  # For gpu change it to cuda
        global g_device
        g_device = self.device
        self.device2 = torch.device('cuda')
        self.params = pickle.load(open(base + '/SOCNAV_V2.prms', 'rb'), fix_imports=True)
        self.params['net'] = self.params['net'].lower()
        self.net = self.params['net']
        print(self.params)
        self.GNNmodel = SELECT_GNN(num_features=self.params['num_feats'],
                                   num_edge_feats=self.params['num_edge_feats'],
                                   n_classes=self.params['n_classes'],
                                   num_hidden=self.params['num_hidden'],
                                   gnn_layers=self.params['gnn_layers'],
                                   cnn_layers=self.params['cnn_layers'],
                                   dropout=self.params['in_drop'],
                                   activation=self.params['nonlinearity'],
                                   final_activation=self.params['final_activation'],
                                   activation_cnn=self.params['nonlinearity_cnn'],
                                   final_activation_cnn=self.params['final_activation_cnn'],
                                   num_channels=self.params['num_channels'],
                                   gnn_type=self.params['net'],
                                   K=10,  # sage filters
                                   num_heads=self.params['heads'],
                                   num_rels=self.params['num_rels'],
                                   num_bases=self.params['num_bases'],
                                   g=None,
                                   residual=self.params['residual'],
                                   aggregator_type=self.params['aggregator_type'],
                                   attn_drop=self.params['attn_drop'],
                                   num_hidden_layers_rgcn=self.params['gnn_layers'],
                                   num_hidden_layers_gat=self.params['gnn_layers'],
                                   num_hidden_layer_pairs=self.params['gnn_layers'],
                                   alpha=self.params['alpha'],
                                   )

        self.GNNmodel.load_state_dict(torch.load(base + '/SOCNAV_V2.tch', map_location=device))
        self.GNNmodel.to(self.device)
        self.GNNmodel.eval()

        if dataset is not None:
            self.test_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate)

    def predictOneGraph(self, g):
        self.test_dataloader = DataLoader(g, batch_size=1, collate_fn=collate)
        logits = self.predict()[0]
        return logits

    def predict(self, g=None):
        if g is not None:
            self.test_dataloader = DataLoader(g, batch_size=len(g), collate_fn=collate_list)

        result = []
        for batch, data in enumerate(self.test_dataloader):
            subgraph, labels = data
            feats = subgraph.ndata['h'].to(self.device)
            if 'he' in subgraph.edata.keys():
                efeats = subgraph.edata['he'].to(self.device)
            else:
                efeats = None

            if self.params['fw'] == 'dgl':
                self.GNNmodel.gnn_object.g = subgraph
                self.GNNmodel.g = subgraph
                for layer in self.GNNmodel.gnn_object.layers:
                    layer.g = subgraph
                if self.params['net'] in ['rgcn']:
                    logits = self.GNNmodel(feats.float(), subgraph.edata['rel_type'].squeeze().to(self.device), None)
                elif self.params['net'] in ['mpnn']:
                    logits = self.GNNmodel(feats.float(), subgraph, efeats.float())
                else:
                    logits = self.GNNmodel(feats.float(), subgraph, None)
            else:
                print('Error!! the graph framework must be dgl')
                sys.exit(1)

            result.append(logits)
        return result

class Human():
    def __init__(self, id, x, y, a, vx, vy, va):
        self.id = id
        self.x = x
        self.y = y
        self.a = a
        self.vx = vx
        self.vy = vy
        self.va = va        


class Object():
    def __init__(self, id, x, y, a, vx, vy, va, size_x, size_y):
        self.id = id
        self.x = x
        self.y = y
        self.a = a
        self.vx = vx
        self.vy = vy
        self.va = va        
        self.size_x = size_x
        self.size_y = size_y

class SNScenario():
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.room = None
        self.humans = []
        self.objects = []
        self.interactions = []
        self.goal = None
        self.command = None

    def add_room(self, sn_room):
        self.room = sn_room

    def add_human(self, sn_human):
        self.humans.append(sn_human)

    def add_object(self, sn_object):
        self.objects.append(sn_object)

    def add_interaction(self, sn_interactions):
        self.interactions.append(sn_interactions)

    def add_goal(self, x, y):
        self.goal=[x, y]
    
    def add_command(self, command):
        self.command = command

    def to_json(self):
        jsonmodel = {}
        jsonmodel['ID'] = "A000000"
        # Adding Room
        jsonmodel['walls'] = self.room
        # Adding humans and objects
        jsonmodel['people'] = []
        jsonmodel['objects'] = []
        for _human in self.humans:
            human = {}
            human['id'] = int(_human.id)
            human['x'] = float(_human.x)
            human['y'] = float(_human.y)
            human['a'] = float(_human.a)
            human['vx'] = float(_human.vx)
            human['vy'] = float(_human.vy)
            human['va'] = float(_human.va)
            jsonmodel['people'].append(human)
        for _object in self.objects:
            Object = {}
            Object['id'] = int(_object.id)
            Object['x'] = float(_object.x)
            Object['y'] = float(_object.y)
            Object['a'] = float(_object.a)
            Object['vx'] = float(_object.vx)
            Object['vy'] = float(_object.vy)
            Object['va'] = float(_object.va)
            Object['size_x'] = float(_object.size_x)
            Object['size_y'] = float(_object.size_y)
            jsonmodel['objects'].append(Object)
        # Adding links
        jsonmodel['interaction'] = []
        for interaction in self.interactions:
            link = {}
            link['dst'] = int(interaction[0])
            link['src'] = int(interaction[1])
            link['relation'] = 'interaction'
            jsonmodel['interaction'].append(link)
        jsonmodel['goal'] = [{'x' : self.goal[0], 'y' : self.goal[1]}]
        jsonmodel['command'] = self.command
        jsonmodel['label_Q1'] = 0
        jsonmodel['label_Q2'] = 0
        jsonmodel['timestamp'] = float(self.timestamp)
        return jsonmodel
