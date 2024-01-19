import copy
from collections import namedtuple
import math

import torch as th
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

N_INTERVALS = 3
FRAMES_INTERVAL = 1.

grid_width = 19  # 30 #18
image_width = 256  # 121 #73
area_width = 700.  # horizontal/vertical distance between two contiguous nodes of the grid. Previously it was taken
# as the spatial area of the grid

threshold_human_wall = 1.5
limit = 9000000000000  # Limit of graphs to load
path_saves = 'saves/'  # This variable is necessary due to a bug in dgl.DGLDataset source code
graphData = namedtuple('graphData', ['src_nodes', 'dst_nodes', 'n_nodes', 'features', 'edge_feats', 'edge_types',
                                     'edge_norms', 'position_by_id', 'typeMap', 'w_segments'])
gridData = namedtuple('graphData', ['src_nodes', 'dst_nodes', 'n_nodes', 'features', 'edge_feats', 'edge_types',
                                     'edge_norms', 'position_by_id', 'typeMap', 'ids', 'w_segments'])

Wall = namedtuple('Wall', ['orientation', 'xpos', 'ypos'])

MAX_ADV = 3.5
MAX_ROT = 4.
MAX_HUMANS = 15


# Transformation matrix
def get_transformation_matrix_for_pose(x, z, angle):
    M = np.zeros((3, 3))
    M[0][0] = +math.cos(-angle)
    M[0][1] = -math.sin(-angle)
    M[0][2] = x
    M[1][0] = +math.sin(-angle)
    M[1][1] = +math.cos(-angle)
    M[1][2] = z
    M[2][2] = 1.0
    return M


#  human to wall distance
def dist_h_w(h, wall):
    if 'xPos' in h.keys():
        hxpos = float(h['xPos']) / 100.
        hypos = float(h['yPos']) / 100.
    else:
        hxpos = float(h['x'])
        hypos = float(h['y'])

    wxpos = float(wall.xpos) / 100.
    wypos = float(wall.ypos) / 100.
    return math.sqrt((hxpos - wxpos) * (hxpos - wxpos) + (hypos - wypos) * (hypos - wypos))


# return de number of grid nodes if a grid is used in the specified alternative
def grid_nodes_number(alt):
    if alt == '2' or alt == '7' or alt == '8':
        return grid_width * grid_width
    else:
        return 0


def central_grid_nodes(alt, r):
    if alt == '7' or alt == '8':
        grid_node_ids = np.zeros((grid_width, grid_width), dtype=int)
        for y in range(grid_width):
            for x in range(grid_width):
                grid_node_ids[x][y] = y * grid_width + x
        central_nodes = closest_grid_nodes(grid_node_ids, area_width, grid_width, r, 0, 0)
        return central_nodes
    else:
        return []


def calculate_relative_position(entity1, entity2):
    x1, y1, a1 = entity1
    x2, y2, a2 = entity2

    ang = a2 - a1
    ang = math.atan2(math.sin(ang), math.cos(ang))

    p = np.array([[x2], [y2], [1.0]], dtype=float)
    M = np.linalg.inv(get_transformation_matrix_for_pose(x1, y1, a1))
    p = M.dot(p)

    return p[0][0], p[1][0], ang


# Calculate the closet node in the grid to a given node by its coordinates
def closest_grid_node(grid_ids, w_a, w_i, x, y):
    c_x = int(round(x / w_a) + (w_i // 2))
    if c_x < 0: c_x = 0
    if c_x >= grid_width: c_x = grid_width - 1
    c_y = int(round(y / w_a) + (w_i // 2))
    if c_y < 0: c_y = 0
    if c_y >= grid_width: c_y = grid_width - 1
    return grid_ids[c_x][c_y]

    # if 0 <= c_x < grid_width and 0 <= c_y < grid_width:
    #     return grid_ids[c_x][c_y]
    # return None


def closest_grid_nodes(grid_ids, w_a, w_i, r, x, y):
    c_x = int(round(x / w_a) + (w_i // 2))
    c_y = int(round(y / w_a) + (w_i // 2))
    cols, rows = (int(math.ceil(r / w_a)), int(math.ceil(r / w_a)))
    rangeC = list(range(-cols, cols + 1))
    rangeR = list(range(-rows, rows + 1))
    p_arr = [[c, r] for c in rangeC for r in rangeR]
    grid_nodes = []
    # r_g = r / w_a
    for p in p_arr:
        g_x, g_y = c_x + p[0], c_y + p[1]
        gw_x, gw_y = (g_x - w_i // 2) * w_a, (g_y - w_i // 2) * w_a
        if math.sqrt((gw_x - x) * (gw_x - x) + (gw_y - y) * (gw_y - y)) <= r:
            # if math.sqrt(p[0] * p[0] + p[1] * p[1]) <= r_g:
            if 0 <= g_x < grid_width and 0 <= g_y < grid_width:
                grid_nodes.append(grid_ids[g_x][g_y])

    return grid_nodes

def generate_static_tables(w_a, w_i, r):
    cols, rows = (int(math.ceil(r / w_a)), int(math.ceil(r / w_a)))
    rangeC = list(range(-cols, cols + 1))
    rangeR = list(range(-rows, rows + 1))
    p_arr = [[c_rel, r_rel] for c_rel in rangeC for r_rel in rangeR]
    return p_arr

def closest_grid_nodes_opt(grid_ids, p_arr, w_a, w_i, r, x, y):
    c_x = int(round(x / w_a) + (w_i // 2))
    c_y = int(round(y / w_a) + (w_i // 2))
    grid_nodes = []
    for p in p_arr:
        g_x, g_y = c_x + p[0], c_y + p[1]
        gw_x, gw_y = (g_x - w_i // 2) * w_a, (g_y - w_i // 2) * w_a
        if math.sqrt((gw_x - x) * (gw_x - x) + (gw_y - y) * (gw_y - y)) <= r:
            if 0 <= g_x < grid_width and 0 <= g_y < grid_width:
                grid_nodes.append(grid_ids[g_x][g_y])

    return grid_nodes


def get_relations(alt):
    rels = None
    if alt == '1':
        rels = {'p_r', 'o_r', 'l_r', 'l_p', 'l_o', 'p_p', 'p_o', 'w_l', 'w_p'}
        # p = person
        # r = robot
        # l = room (lounge)
        # o = object
        # w = wall
        # n = node (generic)
        for e in list(rels):
            rels.add(e[::-1])
        rels.add('self')
        rels = sorted(list(rels))
    elif alt == '2':
        room_set = {'l_p', 'l_o', 'l_w', 'l_g', 'p_p', 'p_o', 'p_g', 'o_g', 'w_g'}
        grid_set = {'g_c', 'g_ri', 'g_le', 'g_u', 'g_d', 'g_uri', 'g_dri', 'g_ule', 'g_dle'}
        # ^
        # |_p = person             g_ri = grid right
        # |_w = wall               g_le = grid left
        # |_l = lounge             g_u = grid up
        # |_o = object             g_d = grid down
        # |_g = grid node
        self_edges_set = {'P', 'O', 'W', 'L'}

        for e in list(room_set):
            room_set.add(e[::-1])
        relations_class = room_set | grid_set | self_edges_set
        rels = sorted(list(relations_class))
    elif alt == '3':
        rels = {'p_r', 'o_r', 'p_p', 'p_o', 'w_r', 't_r', 'w_p'}  # add 'w_w' for links between wall nodes
        # p = person
        # r = room
        # o = object
        # w = wall
        # t = goal
        for e in list(rels):
            rels.add(e[::-1])
        rels.add('self')
        rels = sorted(list(rels))
    elif alt == '4':
        rels = {'o_r', 'g_r'}
        # r = room
        # o = object
        # g = goal
        for e in list(rels):
            rels.add(e[::-1])
        rels.add('self')
        rels = sorted(list(rels))
    elif alt == '5' or alt == '6':
        rels = {'p_r', 't_r'}
        # r = room
        # p = person
        # t = goal (target)
        for e in list(rels):
            rels.add(e[::-1])
        rels.add('self')
        rels = sorted(list(rels))
    elif alt == '7':
        rels = {'p_r', 't_r', 'w_r', 'p_g', 't_g', 'r_g', 'w_g'}
        grid_set = {'g_c', 'g_ri', 'g_le', 'g_u', 'g_d', 'g_uri', 'g_dri', 'g_ule', 'g_dle'}
        # r = room
        # p = person
        # t = goal (target)
        # w = wall
        # g = grid
        for e in list(rels):
            rels.add(e[::-1])
        rels.add('self')
        rels = rels | grid_set
        rels = sorted(list(rels))
    elif alt == '8' or alt == '9':
        rels = {'p_r', 'o_r', 'p_p', 'p_o', 't_r', 'w_r', 'p_g', 'o_g', 't_g', 'r_g', 'w_g'}
        grid_set = {'g_c', 'g_ri', 'g_le', 'g_u', 'g_d', 'g_uri', 'g_dri', 'g_ule', 'g_dle'}
        # r = room
        # p = person
        # o = object
        # t = goal (target)
        # w = wall
        # g = grid
        for e in list(rels):
            rels.add(e[::-1])
        rels.add('self')
        rels = rels | grid_set
        rels = sorted(list(rels))

    num_rels = len(rels)

    return rels, num_rels


def get_features(alt):
    all_features = None
    if alt == '1':
        node_types_one_hot = ['robot', 'human', 'object', 'room', 'wall']
        human_metric_features = ['hum_distance', 'hum_distance2', 'hum_angle_sin', 'hum_angle_cos',
                                 'hum_orientation_sin', 'hum_orientation_cos', 'hum_robot_sin',
                                 'hum_robot_cos']
        object_metric_features = ['obj_distance', 'obj_distance2', 'obj_angle_sin', 'obj_angle_cos',
                                  'obj_orientation_sin', 'obj_orientation_cos']
        room_metric_features = ['room_min_human', 'room_min_human2', 'room_humans', 'room_humans2']
        wall_metric_features = ['wall_distance', 'wall_distance2', 'wall_angle_sin', 'wall_angle_cos',
                                'wall_orientation_sin', 'wall_orientation_cos']
        all_features = node_types_one_hot + human_metric_features + object_metric_features + room_metric_features + \
                       wall_metric_features
    elif alt == '2':
        node_types_one_hot = ['human', 'object', 'room', 'wall', 'grid']
        human_metric_features = ['hum_x_pos', 'hum_y_pos', 'hum_orientation_sin', 'hum_orientation_cos']
        object_metric_features = ['obj_x_pos', 'obj_y_pos', 'obj_orientation_sin', 'obj_orientation_cos']
        room_metric_features = ['room_humans', 'room_humans2']
        wall_metric_features = ['wall_x_pos', 'wall_y_pos', 'wall_orientation_sin', 'wall_orientation_cos']
        grid_metric_features = ['grid_x_pos', 'grid_y_pos']  # , 'flag_inside_room']  # , 'count']
        all_features = node_types_one_hot + human_metric_features + object_metric_features + room_metric_features + \
                       wall_metric_features + grid_metric_features
    elif alt == '3':
        time_one_hot = ['is_t_0', 'is_t_m1', 'is_t_m2']
        # time_sequence_features = ['is_first_frame', 'time_left']
        human_metric_features = ['hum_x_pos', 'hum_y_pos', 'human_a_vel', 'human_x_vel', 'human_y_vel',
                                 'hum_orientation_sin', 'hum_orientation_cos',
                                 'hum_dist', 'hum_inv_dist']
        object_metric_features = ['obj_x_pos', 'obj_y_pos', 'obj_a_vel', 'obj_x_vel', 'obj_y_vel',
                                  'obj_orientation_sin', 'obj_orientation_cos',
                                  'obj_x_size', 'obj_y_size',
                                  'obj_dist', 'obj_inv_dist']
        room_metric_features = ['room_humans', 'room_humans2']
        robot_features = ['robot_adv_vel', 'robot_rot_vel']
        wall_metric_features = ['wall_x_pos', 'wall_y_pos', 'wall_orientation_sin', 'wall_orientation_cos',
                                'wall_dist', 'wall_inv_dist']
        goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
        grid_metric_features = ['grid_x_pos', 'grid_y_pos']
        node_types_one_hot = ['human', 'object', 'room', 'wall', 'goal']
        all_features = node_types_one_hot + time_one_hot + human_metric_features + robot_features + \
                       object_metric_features + room_metric_features + wall_metric_features + goal_metric_features
    elif alt == '4':
        object_metric_features = ['obj_x_pos', 'obj_y_pos', 'obj_a_vel', 'obj_x_vel', 'obj_y_vel',
                                  'obj_orientation_sin', 'obj_orientation_cos',
                                  'obj_x_size', 'obj_y_size',
                                  'obj_dist', 'obj_inv_dist']
        robot_features = ['robot_adv_vel', 'robot_rot_vel']
        goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
        node_types_one_hot = ['object', 'room', 'goal']
        all_features = node_types_one_hot + robot_features + \
                       object_metric_features + goal_metric_features
        if N_INTERVALS > 1:
            time_one_hot = ['is_t_0']
            for i in range(1, N_INTERVALS):
                time_one_hot.append('is_t_m' + str(i))
            all_features += time_one_hot
    elif alt == '5':
        human_metric_features = ['hum_x_pos', 'hum_y_pos', 'human_a_vel', 'human_x_vel', 'human_y_vel',
                                 'hum_orientation_sin', 'hum_orientation_cos',
                                 'hum_dist', 'hum_inv_dist']
        robot_features = ['robot_adv_vel', 'robot_rot_vel']
        goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
        node_types_one_hot = ['human', 'room', 'goal']
        all_features = node_types_one_hot + robot_features + \
                       human_metric_features + goal_metric_features
        if N_INTERVALS > 1:
            time_one_hot = ['is_t_0']
            for i in range(1, N_INTERVALS):
                time_one_hot.append('is_t_m' + str(i))
            all_features += time_one_hot
    elif alt == '6':
        human_metric_features = ['hum_x_pos', 'hum_y_pos', 'human_a_vel', 'human_x_vel', 'human_y_vel',
                                 'hum_orientation_sin', 'hum_orientation_cos',
                                 'hum_dist', 'hum_inv_dist']
        robot_features = ['robot_adv_vel', 'robot_rot_vel']
        goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
        step_features = ['step_fraction']
        all_features_1_instant = robot_features + human_metric_features + goal_metric_features + step_features
        all_features = copy.deepcopy(all_features_1_instant)

        # One hot time features
        time_features = ["t0"]
        for i in range(1, N_INTERVALS):
            all_features += [f + '_t' + str(i) for f in all_features_1_instant]
            time_features.append('t' + str(i))

        all_features += time_features
        node_types_one_hot = ['human', 'room', 'goal']
        all_features += node_types_one_hot

    elif alt == '7':
        human_metric_features = ['hum_x_pos', 'hum_y_pos', 'human_a_vel', 'human_x_vel', 'human_y_vel',
                                 'hum_orientation_sin', 'hum_orientation_cos',
                                 'hum_dist', 'hum_inv_dist']
        wall_metric_features = ['wall_x_pos', 'wall_y_pos', 'wall_orientation_sin', 'wall_orientation_cos',
                                'wall_dist', 'wall_inv_dist']
        robot_features = ['robot_adv_vel', 'robot_rot_vel']
        goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
        step_features = ['step_fraction']
        all_features_1_instant = robot_features + human_metric_features + \
                                 wall_metric_features + goal_metric_features + step_features
        all_features = copy.deepcopy(all_features_1_instant)

        # One hot time features
        time_features = ["t0"]
        for i in range(1, N_INTERVALS):
            all_features += [f + '_t' + str(i) for f in all_features_1_instant]
            time_features.append('t' + str(i))

        all_features += time_features
        node_types_one_hot = ['human', 'room', 'goal', 'wall', 'grid']
        grid_metric_features = ['grid_x_pos', 'grid_y_pos']
        all_features += node_types_one_hot + grid_metric_features

    elif alt == '8' or alt == '9':
        human_metric_features = ['hum_x_pos', 'hum_y_pos', 'human_a_vel', 'human_x_vel', 'human_y_vel',
                                 'hum_orientation_sin', 'hum_orientation_cos',
                                 'hum_dist', 'hum_inv_dist']
        object_metric_features = ['obj_x_pos', 'obj_y_pos', 'obj_a_vel', 'obj_x_vel', 'obj_y_vel',
                                  'obj_orientation_sin', 'obj_orientation_cos',
                                  'obj_x_size', 'obj_y_size',
                                  'obj_dist', 'obj_inv_dist']
        wall_metric_features = ['wall_x_pos', 'wall_y_pos', 'wall_orientation_sin', 'wall_orientation_cos',
                                'wall_dist', 'wall_inv_dist']
        robot_features = ['robot_adv_vel', 'robot_rot_vel']
        goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
        step_features = ['step_fraction']
        all_features_1_instant = robot_features + human_metric_features + object_metric_features + \
                                 wall_metric_features + goal_metric_features + step_features
        all_features = copy.deepcopy(all_features_1_instant)

        # One hot time features
        time_features = ["t0"]
        for i in range(1, N_INTERVALS):
            all_features += [f + '_t' + str(i) for f in all_features_1_instant]
            time_features.append('t' + str(i))

        all_features += time_features
        node_types_one_hot = ['human', 'object', 'room', 'goal', 'wall', 'grid']
        grid_metric_features = ['grid_x_pos', 'grid_y_pos']
        all_features += node_types_one_hot + grid_metric_features

    feature_dimensions = len(all_features)

    return all_features, feature_dimensions


def get_edge_features(alt):
    if alt == '9':
        # Define features
        relative_position = ['x', 'y', 'orientation']
        time_to_collision = ['t_collision']
        all_features_1_instant = relative_position + time_to_collision
        all_features = copy.deepcopy(all_features_1_instant)

        # One hot time features
        time_features = ["t0"]
        for i in range(1, N_INTERVALS):
            all_features += [f + '_t' + str(i) for f in all_features_1_instant]
            time_features.append('t' + str(i))

        rels, _ = get_relations(alt)
        edge_types_one_hot = ['wandering_human_interacting', 'two_static_person_talking', 'human_laptop_interaction',
                              'human_plant_interaction']
        all_features += time_features + rels + edge_types_one_hot + ['delta_x', 'delta_y']
        n_features = len(all_features)
    else:
        all_features, num_rels = get_relations(alt)
        n_features = num_rels + 4


    return all_features, n_features


# Generate data for a grid of nodes
def generate_grid_graph_data(alt='2'):
    # Define variables for edge types and relations
    grid_rels, num_rels = get_relations(alt)
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge

    # Grid properties
    connectivity = 8  # Connections of each node
    node_ids = np.zeros((grid_width, grid_width), dtype=int)  # Array to store the IDs of each node
    typeMap = dict()
    coordinates_gridGraph = dict()  # Dict to store the spatial coordinates of each node
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # Feature dimensions
    all_features, n_features = get_features(alt)

    # Compute the number of nodes and initialize feature vectors
    all_edge_features, n_edge_features = get_edge_features(alt)
    n_nodes = grid_width ** 2
    features_gridGraph = th.zeros(n_nodes, n_features)
    edge_feats_list = []

    max_used_id = -1
    for y in range(grid_width):
        for x in range(grid_width):
            max_used_id += 1
            node_id = max_used_id
            node_ids[x][y] = node_id

            # Self edges
            src_nodes.append(node_id)
            dst_nodes.append(node_id)
            edge_types.append(grid_rels.index('g_c'))
            edge_norms.append([1.])
            edge_features = th.zeros(n_edge_features)
            edge_features[all_edge_features.index('g_c')] = 1
            if alt != '9':
                edge_features[-1] = 0
            edge_feats_list.append(edge_features)

            if x < grid_width - 1:
                src_nodes.append(node_id)
                dst_nodes.append(node_id + 1)
                edge_types.append(grid_rels.index('g_ri'))
                edge_norms.append([1.])
                edge_features = th.zeros(n_edge_features)
                edge_features[all_edge_features.index('g_ri')] = 1
                if alt == '9':
                    edge_features[all_edge_features.index('delta_x')] = 1.
                    edge_features[all_edge_features.index('delta_y')] = 0.
                else:
                    edge_features[-1] = 1.
                edge_feats_list.append(edge_features)

                if connectivity == 8 and y > 0:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id - grid_width + 1)
                    edge_types.append(grid_rels.index('g_uri'))
                    edge_norms.append([1.])
                    edge_features = th.zeros(n_edge_features)
                    edge_features[all_edge_features.index('g_uri')] = 1
                    if alt == '9':
                        edge_features[all_edge_features.index('delta_x')] = 1.
                        edge_features[all_edge_features.index('delta_y')] = 1.
                    else:
                        edge_features[-1] = math.sqrt(2.)
                    edge_feats_list.append(edge_features)

            if x > 0:
                src_nodes.append(node_id)
                dst_nodes.append(node_id - 1)
                edge_types.append(grid_rels.index('g_le'))
                edge_norms.append([1.])
                edge_features = th.zeros(n_edge_features)
                edge_features[all_edge_features.index('g_le')] = 1
                if alt == '9':
                    edge_features[all_edge_features.index('delta_x')] = -1.
                    edge_features[all_edge_features.index('delta_y')] = 0.
                else:
                    edge_features[-1] = 1.
                edge_feats_list.append(edge_features)

                if connectivity == 8 and y < grid_width - 1:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id + grid_width - 1)
                    edge_types.append(grid_rels.index('g_dle'))
                    edge_norms.append([1.])
                    edge_features = th.zeros(n_edge_features)
                    edge_features[all_edge_features.index('g_dle')] = 1
                    if alt == '9':
                        edge_features[all_edge_features.index('delta_x')] = -1.
                        edge_features[all_edge_features.index('delta_y')] = -1.
                    else:
                        edge_features[-1] = math.sqrt(2.)
                    edge_feats_list.append(edge_features)

            if y < grid_width - 1:
                src_nodes.append(node_id)
                dst_nodes.append(node_id + grid_width)
                edge_types.append(grid_rels.index('g_d'))
                edge_norms.append([1.])
                edge_features = th.zeros(n_edge_features)
                edge_features[all_edge_features.index('g_d')] = 1
                if alt == '9':
                    edge_features[all_edge_features.index('delta_x')] = 0.
                    edge_features[all_edge_features.index('delta_y')] = -1.
                else:
                    edge_features[-1] = 1.
                edge_feats_list.append(edge_features)

                if connectivity == 8 and x < grid_width - 1:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id + grid_width + 1)
                    edge_types.append(grid_rels.index('g_dri'))
                    edge_norms.append([1.])
                    edge_features = th.zeros(n_edge_features)
                    edge_features[all_edge_features.index('g_dri')] = 1
                    if alt == '9':
                        edge_features[all_edge_features.index('delta_x')] = 1.
                        edge_features[all_edge_features.index('delta_y')] = -1.
                    else:
                        edge_features[-1] = math.sqrt(2.)
                    edge_feats_list.append(edge_features)

            if y > 0:
                src_nodes.append(node_id)
                dst_nodes.append(node_id - grid_width)
                edge_types.append(grid_rels.index('g_u'))
                edge_norms.append([1.])
                edge_features = th.zeros(n_edge_features)
                edge_features[all_edge_features.index('g_u')] = 1
                if alt == '9':
                    edge_features[all_edge_features.index('delta_x')] = 0.
                    edge_features[all_edge_features.index('delta_y')] = 1.
                else:
                    edge_features[-1] = 1.
                edge_feats_list.append(edge_features)

                if connectivity == 8 and x > 0:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id - grid_width - 1)
                    edge_types.append(grid_rels.index('g_ule'))
                    edge_norms.append([1.])
                    edge_features = th.zeros(n_edge_features)
                    edge_features[all_edge_features.index('g_ule')] = 1
                    if alt == '9':
                        edge_features[all_edge_features.index('delta_x')] = -1.
                        edge_features[all_edge_features.index('delta_y')] = 1.
                    else:
                        edge_features[-1] = math.sqrt(2.)
                    edge_feats_list.append(edge_features)

            typeMap[node_id] = 'g'  # 'g' for 'grid'
            x_pos = (x - grid_width // 2) * area_width
            y_pos = (y - grid_width // 2) * area_width
            # x_pos = (-area_width / 2. + (x + 0.5) * (area_width / grid_width))
            # y_pos = (-area_width / 2. + (y + 0.5) * (area_width / grid_width))
            features_gridGraph[node_id, all_features.index('grid')] = 1
            features_gridGraph[node_id, all_features.index('grid_x_pos')] = x_pos / 10000
            features_gridGraph[node_id, all_features.index('grid_y_pos')] = y_pos / 10000

            coordinates_gridGraph[node_id] = [x_pos / 10000, y_pos / 10000]

    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)
    edge_feats = th.stack(edge_feats_list)

    return src_nodes, dst_nodes, n_nodes, features_gridGraph, edge_feats, edge_types, edge_norms, coordinates_gridGraph, typeMap, \
           node_ids, []


def generate_walls_information(data, w_segments):
    # Compute data for walls
    walls = []
    i_w = 0
    wall_index = []
    for wall_segment in data['walls']:
        p1 = np.array([wall_segment["x1"], wall_segment["y1"]]) * 100
        p2 = np.array([wall_segment["x2"], wall_segment["y2"]]) * 100
        dist = np.linalg.norm(p1 - p2)
        if i_w >= len(w_segments):
            iters = int(dist / 249) + 1
            w_segments.append(iters)
        if w_segments[i_w] > 1:  # WE NEED TO CHECK THIS PART
            v = (p2 - p1) / w_segments[i_w]
            for i in range(w_segments[i_w]):
                pa = p1 + v * i
                pb = p1 + v * (i + 1)
                inc2 = pb - pa
                midsp = (pa + pb) / 2
                walls.append(Wall(math.atan2(inc2[0], inc2[1]), midsp[0], midsp[1]))
                wall_index.append(i_w)
        else:
            inc = p2 - p1
            midp = (p2 + p1) / 2
            walls.append(Wall(math.atan2(inc[0], inc[1]), midp[0], midp[1]))
            wall_index.append(i_w)
        i_w += 1
    return walls, w_segments, wall_index
