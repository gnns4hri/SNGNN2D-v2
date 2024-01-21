import os
import sys
import json

import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info

import cv2

from .alternatives import *

#################################################################
# Class to load the dataset
#################################################################


class SocNavDataset(DGLDataset):
    grid_data = gridData(*generate_grid_graph_data('8'))
    default_alt = '8'
    p_arr = generate_static_tables(area_width, grid_width, area_width)

    def __init__(self, path, alt, net, mode='train', raw_dir='data/', prev_graph = None, init_line=-1, end_line=-1, loc_limit=limit,
                 force_reload=False, verbose=False, debug=False, device = 'cpu'):
        if type(path) is str:
            self.path = raw_dir + path
        else:
            self.path = path
        self.mode = mode
        self.net = net
        self.alt = alt
        self.init_line = init_line
        self.end_line = end_line
        self.prev_graph = prev_graph
        self.graphs = []
        self.labels = []
        self.data = dict()
        # self.grid_data = None
        self.data['typemaps'] = []
        self.data['coordinates'] = []
        self.data['identifiers'] = []
        self.data['wall_segments'] = []
        self.debug = debug
        self.limit = loc_limit
        self.force_reload = force_reload
        self.edge_features = (self.net == 'mpnn')


        if self.mode == 'test':
            self.force_reload = True

        # Define device. GPU if it is available
        self.device = device #'cuda' #'cpu'

        if self.debug:
            self.limit = 1 + (0 if init_line == -1 else init_line)
        
        if self.default_alt != self.alt:
            self.default_alt = self.alt
            if self.alt == '7' or self.alt == '8':
                self.grid_data = gridData(*generate_grid_graph_data(self.alt))

        if self.alt == '1':
            self.dataloader = initializeAlt1
        elif self.alt == '2':
            self.dataloader = initializeAlt2
        elif self.alt == '3':
            self.dataloader = initializeAlt3
        elif self.alt == '4':
            self.dataloader = initializeAlt4
        elif self.alt == '5':
            self.dataloader = initializeAlt5
        elif self.alt == '6':
            self.dataloader = initializeAlt6
        elif self.alt == '7':
            self.dataloader = initializeAlt6
        elif self.alt == '8' or self.alt == '9':
            self.dataloader = initializeAlt8

        else:
            print('Introduce a valid initialize alternative')
            sys.exit(-1)

        super(SocNavDataset, self).__init__("SocNav", raw_dir=raw_dir, force_reload=self.force_reload, verbose=verbose)

    def get_dataset_name(self):
        graphs_path = 'graphs_' + self.mode + '_alt_' + self.alt + '_efeats_' + str(self.edge_features) + '_s_' + str(limit) + '.bin'
        info_path = 'info_' + self.mode + '_alt_' + self.alt + '_efeats_' + str(self.edge_features) + '_s_' + str(limit) + '.pkl'
        return graphs_path, info_path

    def generate_final_graph(self, raw_data):
        # time_0 = time.time()
        rels, num_rels = get_relations(self.alt)
        all_edge_features, n_edge_features = get_edge_features(self.alt)
        if self.alt == '8':
            room_graph_data = graphData(*self.dataloader(raw_data, self.alt, [], self.edge_features))
        else:
            room_graph_data = graphData(*self.dataloader(raw_data, self.alt, []))

        if self.grid_data is not None:
            # Merge room and grid graph
            src_nodes = th.cat([self.grid_data.src_nodes,(room_graph_data.src_nodes + self.grid_data.n_nodes)], dim=0)
            dst_nodes = th.cat([self.grid_data.dst_nodes, (room_graph_data.dst_nodes + self.grid_data.n_nodes)], dim=0)
            edge_types = th.cat([self.grid_data.edge_types, room_graph_data.edge_types], dim=0)
            edge_norms = th.cat([self.grid_data.edge_norms, room_graph_data.edge_norms], dim=0)
            if self.edge_features:
                edge_feats = th.cat([self.grid_data.edge_feats, room_graph_data.edge_feats], dim=0)
                edge_feats_list = []

            # Link each node in the room graph to the correspondent grid graph.
            for r_n_id in range(0, room_graph_data.n_nodes):
                r_n_type = room_graph_data.typeMap[r_n_id]
                # if self.alt == '9':
                #     x, y, _ = room_graph_data.position_by_id[r_n_id]
                # else:
                x, y = room_graph_data.position_by_id[r_n_id]
                grid_distance = area_width / grid_width
                closest_grid_nodes_id = closest_grid_nodes_opt(self.grid_data.ids, self.p_arr, area_width, grid_width, area_width, x * 10000, y * 10000)
                # closest_grid_nodes_id = closest_grid_nodes(self.grid_data.ids, area_width, grid_width, area_width, x * 10000, y * 10000)
                
                for g_id in closest_grid_nodes_id:
                    if g_id is not None:
                        x_g, y_g = self.grid_data.position_by_id[g_id]
                        src_nodes = th.cat([src_nodes, th.tensor([g_id], dtype=th.int32)], dim=0)
                        dst_nodes = th.cat([dst_nodes, th.tensor([r_n_id + self.grid_data.n_nodes], dtype=th.int32)], dim=0)
                        edge_types = th.cat([edge_types, th.IntTensor([rels.index('g_' + r_n_type)])], dim=0)
                        edge_norms = th.cat([edge_norms, th.Tensor([[1.]])])
                        if self.edge_features:
                            new_edge_features = th.zeros(n_edge_features)
                            new_edge_features[all_edge_features.index('g_'+ r_n_type)] = 1
                            # if self.alt == '9':
                            #     new_edge_features[all_edge_features.index('delta_x')] = (x_g - x)/grid_distance
                            #     new_edge_features[all_edge_features.index('delta_y')] = (y_g - y)/grid_distance
                            edge_feats_list.append(new_edge_features)


                        src_nodes = th.cat([src_nodes, th.tensor([r_n_id + self.grid_data.n_nodes], dtype=th.int32)], dim=0)
                        dst_nodes = th.cat([dst_nodes, th.tensor([g_id], dtype=th.int32)], dim=0)
                        edge_types = th.cat([edge_types, th.IntTensor([rels.index(r_n_type + '_g')])], dim=0)
                        edge_norms = th.cat([edge_norms, th.Tensor([[1.]])])
                        if self.edge_features:
                            new_edge_features = th.zeros(n_edge_features)
                            new_edge_features[all_edge_features.index(r_n_type + '_g')] = 1
                            # if self.alt == '9':
                            #     new_edge_features[all_edge_features.index('delta_x')] = (x - x_g)/grid_distance
                            #     new_edge_features[all_edge_features.index('delta_y')] = (y - y_g)/grid_distance
                            edge_feats_list.append(new_edge_features)


            # Compute typemaps, coordinates, number of nodes, features and labels for the merged graph.
            n_nodes = room_graph_data.n_nodes + self.grid_data.n_nodes
            typeMapRoomShift = dict()
            coordinates_roomShift = dict()
            if self.edge_features:
                edge_feats_list = th.stack(edge_feats_list)
                edge_feats = th.cat([edge_feats, edge_feats_list], dim=0)

            for key in room_graph_data.typeMap:
                typeMapRoomShift[key + len(self.grid_data.typeMap)] = room_graph_data.typeMap[key]
                coordinates_roomShift[key + len(self.grid_data.position_by_id)] = room_graph_data.position_by_id[key]

            position_by_id = {**self.grid_data.position_by_id, **coordinates_roomShift}
            typeMap = {**self.grid_data.typeMap, **typeMapRoomShift}
            features = th.cat([self.grid_data.features, room_graph_data.features], dim=0)
        else:
            src_nodes, dst_nodes, n_nodes, features, edge_feats, edge_types, edge_norms, \
            position_by_id, typeMap, wall_segments = room_graph_data

        # time_2 = time.time()
        # print('room graph', time_1-time_0, 'grid graph', time_2-time_1, 'n. nodes', n_nodes, 'room nodes', room_graph_data.n_nodes)
        self.data['typemaps'].append(typeMap)
        self.data['coordinates'].append(position_by_id)
        self.data['identifiers'].append(raw_data[0]['ID'])

        try:
            final_graph = dgl.graph((src_nodes, dst_nodes), num_nodes=n_nodes, idtype=th.int32, device=self.device)
            final_graph.ndata['h'] = features.to(self.device)
            edge_types = edge_types.to(self.device, dtype=th.long)
            edge_norms = edge_norms.to(self.device, dtype=th.float32)
            if self.edge_features:
                edge_feats = edge_feats.to(self.device, dtype=th.float32)
                final_graph.edata.update({'rel_type': edge_types, 'norm': edge_norms, 'he': edge_feats})
            else:
                final_graph.edata.update({'rel_type': edge_types, 'norm': edge_norms})
            return final_graph
        except Exception:
            raise

    def load_one_graph(self, data):
        # For alternatives 6, 7, 8 and 9 do not create several-instants graphs
        if self.alt == '6'or self.alt == '7' or self.alt == '8' or self.alt == '9':
            graph_data = graphData(*self.dataloader(data))
            src_nodes, dst_nodes, n_nodes, feats, edge_feats, edge_types, edge_norms, coordinates, typeMap, w_segments = graph_data
        else:
            graph_data = graphData(*self.dataloader(data[0]))
            w_segments = graph_data.w_segments
            graphs_in_interval = [graph_data]
            frames_in_interval = [data[0]]
            for frame in data[1:]:
                if len(graphs_in_interval) == N_INTERVALS:
                    break
                if math.fabs(frame['timestamp'] - frames_in_interval[-1][
                    'timestamp']) < FRAMES_INTERVAL:  # Truncated to N seconds
                    continue
                graphs_in_interval.append(graphData(*self.dataloader(frame, w_segments)))
                frames_in_interval.append(frame)

            src_nodes, dst_nodes, edge_types, edge_norms, n_nodes, feats, edge_feats, typeMap, coordinates = \
                self.merge_graphs(graphs_in_interval)
        try:
            # Create merged graph:
            final_graph = dgl.graph((src_nodes, dst_nodes),
                                    num_nodes=n_nodes,
                                    idtype=th.int32, device=self.device)

            # Add merged features and update edge labels:
            final_graph.ndata['h'] = feats.to(self.device)
            final_graph.edata.update({'rel_type': edge_types, 'norm': edge_norms, 'he': edge_feats})

            # Append final data
            self.graphs.append(final_graph)
            self.data['typemaps'].append(typeMap)
            self.data['coordinates'].append(coordinates)
            self.data['identifiers'].append(data[0]['ID'])
            self.data['wall_segments'].append(w_segments)

        except Exception:
            print("Error loading one graph")
            raise


    def merge_graphs(self, graphs_in_interval):
        all_features, n_features = get_features(self.alt)
        new_features = ['is_t_0', 'is_t_m1', 'is_t_m2']
        f_list = []
        src_list = []
        dst_list = []
        edge_types_list = []
        edge_norms_list = []
        edge_feats_list = []
        typeMap = dict()
        coordinates = dict()
        n_nodes = 0
        rels, num_rels = get_relations(self.alt)
        g_i = 0
        offset = graphs_in_interval[0].n_nodes
        for g in graphs_in_interval:
            # Shift IDs of the typemap and coordinates lists
            for key in g.typeMap:
                typeMap[key + (offset * g_i)] = g.typeMap[key]
                coordinates[key + (offset * g_i)] = g.position_by_id[key]
            n_nodes += g.n_nodes
            f_list.append(g.features)
            edge_feats_list.append(g.edge_feats)
            # Add temporal edges
            src_list.append(g.src_nodes + (offset * g_i))
            dst_list.append(g.dst_nodes + (offset * g_i))
            edge_types_list.append(g.edge_types)
            edge_norms_list.append(g.edge_norms)
            if g_i > 0:
                # Temporal connections and edges labels
                new_src_list = []
                new_dst_list = []
                new_etypes_list = []
                new_enorms_list = []
                new_edge_feats_list = []
                for node in range(g.n_nodes):
                    new_src_list.append(node + offset * (g_i - 1))
                    new_dst_list.append(node + offset * g_i)
                    new_etypes_list.append(num_rels + (g_i - 1) * 2)
                    new_enorms_list.append([1.])

                    new_src_list.append(node + offset * g_i)
                    new_dst_list.append(node + offset * (g_i - 1))
                    new_etypes_list.append(num_rels + (g_i - 1) * 2 + 1)
                    new_enorms_list.append([1.])

                    # Edge features
                    edge_features = th.zeros(num_rels + 4)
                    edge_features[num_rels + (g_i - 1) * 2] = 1
                    edge_features[-1] = 0
                    new_edge_feats_list.append(edge_features)

                    edge_features = th.zeros(num_rels + 4)
                    edge_features[num_rels + (g_i - 1) * 2 + 1] = 1
                    edge_features[-1] = 0
                    new_edge_feats_list.append(edge_features)

                new_edge_feats = th.stack(new_edge_feats_list)

                src_list.append(th.IntTensor(new_src_list))
                dst_list.append(th.IntTensor(new_dst_list))
                edge_types_list.append(th.IntTensor(new_etypes_list))
                edge_norms_list.append(th.Tensor(new_enorms_list))
                edge_feats_list.append(new_edge_feats)
            if N_INTERVALS>1:
                for f in new_features:
                    if g_i == new_features.index(f):
                        g.features[:, all_features.index(f)] = 1
                    else:
                        g.features[:, all_features.index(f)] = 0
            g_i += 1

        src_nodes = th.cat(src_list, dim=0)
        dst_nodes = th.cat(dst_list, dim=0)
        edge_types = th.cat(edge_types_list, dim=0)
        edge_norms = th.cat(edge_norms_list, dim=0)
        edge_feats = th.cat(edge_feats_list, dim=0)
        feats = th.cat(f_list, dim=0)

        return src_nodes, dst_nodes, edge_types, edge_norms, n_nodes, feats, edge_feats, typeMap, coordinates

    #################################################################
    # Implementation of abstract methods
    #################################################################

    def download(self):
        # No need to download any data
        pass

    def process(self):
        if type(self.path) is str and self.path.endswith('.json'):
            linen = -1
            for line in open(self.path).readlines():
                if linen % 1000 == 0:
                    print(linen)

                if linen + 1 >= self.limit:
                    print('Stop including more samples to speed up dataset loading')
                    break
                linen += 1
                if self.init_line >= 0 and linen < self.init_line:
                    continue
                if linen > self.end_line >= 0:
                    continue

                raw_data = json.loads(line)
                final_graph = self.generate_final_graph(raw_data)
                self.graphs.append(final_graph)
            
            self.labels.append(np.zeros((1, image_width, image_width)))

            self.labels = th.tensor(self.labels, dtype=th.float32)

        elif type(self.path) is str and self.path.endswith('.txt'):
            linen = -1
            print(self.path)
            with open(self.path) as set_file:
                ds_files = set_file.read().splitlines()

            print("number of files for ", self.path, len(ds_files))

            for ds in ds_files:
                if linen % 1000 == 0:
                    print(linen)
                try:
                    with open(ds) as json_file:
                        data = json.load(json_file)
                    data.reverse()
                except Exception:
                    print("File not found --> ", ds)
                    continue
                if self.alt != '7' and self.alt != '8':
                    self.load_one_graph(data)
                else:
                    final_graph = self.generate_final_graph(data)
                    self.graphs.append(final_graph)

                if self.mode != 'test':
                    img_file = ds.split('.')[0] + '__Q1.png'
                    # label = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2GRAY).astype(float) / 255
                    label = (cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2GRAY).astype(float) / 255.) * 2. - 1
                    label = cv2.resize(label, (image_width, image_width), interpolation=cv2.INTER_CUBIC)
                    label = label.reshape((1, image_width, image_width))
                else:
                    label = np.zeros((1, image_width, image_width))
                self.labels.append(label)

                if linen + 1 >= limit:
                    print('Stop including more samples to speed up dataset loading')
                    break
                linen += 1


            self.labels = th.tensor(self.labels, dtype=th.float32)

        elif type(self.path) == list and len(self.path) >= 1:
            # print('generating graph')
            if self.prev_graph is None:
                if self.alt != '7' and self.alt != '8':
                    self.load_one_graph(self.path)
                else:
                    final_graph = self.generate_final_graph(self.path)
                    self.graphs.append(final_graph)
            else:
                num_instants = sum(map(('r').__eq__, self.prev_graph.data['typemaps'][0].values()))
                if num_instants==N_INTERVALS:
                    self.generate_from_previous_graph(self.path)
                else:
                    self.load_one_graph(self.path)
            self.labels.append(np.zeros((1, image_width, image_width)))        
            self.labels = th.tensor(np.array(self.labels), dtype=th.float32)
        elif type(self.path) == list and type(self.path[0]) == str:
            raw_data = json.loads(self.path)
            final_graph = self.generate_final_graph(raw_data)
            self.graphs.append(final_graph)
            self.labels.append(np.zeros((1, image_width, image_width)))
            self.labels = th.tensor(self.labels, dtype=th.float32)
        else:
            final_graph = self.generate_final_graph(self.path)
            self.graphs.append(final_graph)
            self.labels.append(np.zeros((1, image_width, image_width)))
            self.labels = th.tensor(self.labels, dtype=th.float32)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        if self.debug:
            return
        # Generate paths
        graphs_path, info_path = tuple((path_saves + x) for x in self.get_dataset_name())
        os.makedirs(os.path.dirname(path_saves), exist_ok=True)

        # Save graphs
        save_graphs(graphs_path, self.graphs, {'labels': self.labels})

        # Save additional info
        save_info(info_path, {'typemaps': self.data['typemaps'],
                              'coordinates': self.data['coordinates'],
                              'identifiers': self.data['identifiers']})

    def load(self):
        # Generate paths
        graphs_path, info_path = tuple((path_saves + x) for x in self.get_dataset_name())
        # Load graphs
        self.graphs, label_dict = load_graphs(graphs_path)
        self.labels = label_dict['labels']

        # Load info
        self.data['typemaps'] = load_info(info_path)['typemaps']
        self.data['coordinates'] = load_info(info_path)['coordinates']
        self.data['identifiers'] = load_info(info_path)['identifiers']

    def has_cache(self):
        # Generate paths
        graphs_path, info_path = tuple((path_saves + x) for x in self.get_dataset_name())
        if self.debug:
            return False
        return os.path.exists(graphs_path) and os.path.exists(info_path)
