
from .alternatives_utils import *


#################################################################
# Different initialize alternatives:
#################################################################

def initializeAlt1(data):
    # Initialize variables
    rels, num_rels = get_relations('1')
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge
    max_used_id = 0  # Initialise id counter (0 for the robot)
    closest_human_distance = -1  # Compute closest human distance

    # Compute data for walls
    Wall = namedtuple('Wall', ['dist', 'orientation', 'angle', 'xpos', 'ypos'])
    walls = []
    for wall_index in range(len(data['room']) - 1):
        p1 = np.array(data['room'][wall_index + 0])
        p2 = np.array(data['room'][wall_index + 1])
        dist = np.linalg.norm(p1 - p2)
        iters = int(dist / 400) + 1
        if iters > 1:
            v = (p2 - p1) / iters
            for i in range(iters):
                pa = p1 + v * i
                pb = p1 + v * (i + 1)
                inc2 = pb - pa
                midsp = (pa + pb) / 2
                walls.append(
                    Wall(np.linalg.norm(midsp) / 100., math.atan2(inc2[0], inc2[1]), math.atan2(midsp[0], midsp[1]),
                         midsp[0], midsp[1]))
        else:
            inc = p2 - p1
            midp = (p2 + p1) / 2
            walls.append(
                Wall(np.linalg.norm(inc / 2) / 100., math.atan2(inc[0], inc[1]), math.atan2(midp[0], midp[1]),
                     midp[0], midp[1]))

    # Compute the number of nodes
    # one for the robot + room walls      + humans               + objects          + room(global node)
    n_nodes = 1 + len(walls) + len(data['humans']) + len(data['objects']) + 1

    # Feature dimensions
    all_features, n_features = get_features('1')
    features = th.zeros(n_nodes, n_features)

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes


    # robot (id 0)
    robot_id = 0
    typeMap[robot_id] = 'r'  # 'r' for 'robot'
    features[robot_id, all_features.index('robot')] = 1.

    # humans
    for h in data['humans']:
        src_nodes.append(h['id'])
        dst_nodes.append(robot_id)
        edge_types.append(rels.index('p_r'))
        edge_norms.append([1. / len(data['humans'])])

        src_nodes.append(robot_id)
        dst_nodes.append(h['id'])
        edge_types.append(rels.index('r_p'))
        edge_norms.append([1.])

        typeMap[h['id']] = 'p'  # 'p' for 'person'
        max_used_id = max(h['id'], max_used_id)
        xpos = float(h['xPos']) / 100.
        ypos = float(h['yPos']) / 100.

        position_by_id[h['id']] = [xpos, ypos]
        distance = math.sqrt(xpos * xpos + ypos * ypos)
        angle = math.atan2(xpos, ypos)
        orientation = float(h['orientation']) / 180. * math.pi
        while orientation > math.pi: orientation -= 2. * math.pi
        while orientation < -math.pi: orientation += 2. * math.pi
        if orientation > math.pi:
            orientation -= math.pi
        elif orientation < -math.pi:
            orientation += math.pi
        # Compute point of view from humans
        if angle > 0:
            angle_hum = (angle - math.pi) - orientation
        else:
            angle_hum = (math.pi + angle) - orientation

        # print(str(math.degrees(angle)) + ' ' + str(math.degrees(orientation)) + ' ' + str(math.degrees(angle_hum)))
        features[h['id'], all_features.index('human')] = 1.
        features[h['id'], all_features.index('hum_distance')] = distance
        features[h['id'], all_features.index('hum_distance2')] = distance * distance
        features[h['id'], all_features.index('hum_angle_sin')] = math.sin(angle)
        features[h['id'], all_features.index('hum_angle_cos')] = math.cos(angle)
        features[h['id'], all_features.index('hum_orientation_sin')] = math.sin(orientation)
        features[h['id'], all_features.index('hum_orientation_cos')] = math.cos(orientation)
        features[h['id'], all_features.index('hum_robot_sin')] = math.sin(angle_hum)
        features[h['id'], all_features.index('hum_robot_cos')] = math.cos(angle_hum)
        if closest_human_distance < 0 or closest_human_distance > distance:
            closest_human_distance = distance

    # objects
    for o in data['objects']:
        src_nodes.append(o['id'])
        dst_nodes.append(robot_id)
        edge_types.append(rels.index('o_r'))
        edge_norms.append([1. / len(data['objects'])])

        src_nodes.append(robot_id)
        dst_nodes.append(o['id'])
        edge_types.append(rels.index('r_p'))
        edge_norms.append([1.])

        typeMap[o['id']] = 'o'  # 'o' for 'object'
        max_used_id = max(o['id'], max_used_id)
        xpos = float(o['xPos']) / 100.
        ypos = float(o['yPos']) / 100.

        position_by_id[o['id']] = [xpos, ypos]
        distance = math.sqrt(xpos * xpos + ypos * ypos)
        angle = math.atan2(xpos, ypos)
        orientation = float(o['orientation']) / 180. * math.pi
        while orientation > math.pi: orientation -= 2. * math.pi
        while orientation < -math.pi: orientation += 2. * math.pi
        features[o['id'], all_features.index('object')] = 1
        features[o['id'], all_features.index('obj_distance')] = distance
        features[o['id'], all_features.index('obj_distance2')] = distance * distance
        features[o['id'], all_features.index('obj_angle_sin')] = math.sin(angle)
        features[o['id'], all_features.index('obj_angle_cos')] = math.cos(angle)
        features[o['id'], all_features.index('obj_orientation_sin')] = math.sin(orientation)
        features[o['id'], all_features.index('obj_orientation_cos')] = math.cos(orientation)

    # Room (Global node)
    max_used_id += 1
    room_id = max_used_id
    # print('Room will be {}'.format(room_id))
    typeMap[room_id] = 'l'  # 'l' for 'room' (lounge)
    features[room_id, all_features.index('room')] = 1.
    features[room_id, all_features.index('room_min_human')] = closest_human_distance
    features[room_id, all_features.index('room_min_human2')] = closest_human_distance * closest_human_distance
    features[room_id, all_features.index('room_humans')] = len(data['humans'])
    features[room_id, all_features.index('room_humans2')] = len(data['humans']) * len(data['humans'])

    # walls
    wids = dict()
    for wall in walls:
        max_used_id += 1
        wall_id = max_used_id
        wids[wall] = wall_id
        typeMap[wall_id] = 'w'  # 'w' for 'walls'

        src_nodes.append(wall_id)
        dst_nodes.append(room_id)
        edge_types.append(rels.index('w_l'))
        edge_norms.append([1. / len(walls)])

        src_nodes.append(room_id)
        dst_nodes.append(wall_id)
        edge_types.append(rels.index('l_w'))
        edge_norms.append([1.])

        position_by_id[wall_id] = [wall.xpos / 100., wall.ypos / 100.]
        features[wall_id, all_features.index('wall')] = 1.
        features[wall_id, all_features.index('wall_distance')] = wall.dist
        features[wall_id, all_features.index('wall_distance2')] = wall.dist * wall.dist
        features[wall_id, all_features.index('wall_angle_sin')] = math.sin(wall.angle)
        features[wall_id, all_features.index('wall_angle_cos')] = math.cos(wall.angle)
        features[wall_id, all_features.index('wall_orientation_sin')] = math.sin(wall.orientation)
        features[wall_id, all_features.index('wall_orientation_cos')] = math.cos(wall.orientation)

    for h in data['humans']:
        number = 0
        for wall in walls:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                number -= - 1
        for wall in walls:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                src_nodes.append(wids[wall])
                dst_nodes.append(h['id'])
                edge_types.append(rels.index('w_p'))
                edge_norms.append([1. / number])

    for wall in walls:
        number = 0
        for h in data['humans']:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                number -= - 1
        for h in data['humans']:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                src_nodes.append(h['id'])
                dst_nodes.append(wids[wall])
                edge_types.append(rels.index('p_w'))
                edge_norms.append([1. / number])

    # interaction links
    for link in data['links']:
        typeLdir = typeMap[link[0]] + '_' + typeMap[link[1]]
        typeLinv = typeMap[link[1]] + '_' + typeMap[link[0]]

        src_nodes.append(link[0])
        dst_nodes.append(link[1])
        edge_types.append(rels.index(typeLdir))
        edge_norms.append([1.])

        src_nodes.append(link[1])
        dst_nodes.append(link[0])
        edge_types.append(rels.index(typeLinv))
        edge_norms.append([1.])

    # Edges for the room node (Global Node)
    for node_id in range(n_nodes):
        typeLdir = typeMap[room_id] + '_' + typeMap[node_id]
        typeLinv = typeMap[node_id] + '_' + typeMap[room_id]
        if node_id == room_id:
            continue

        src_nodes.append(room_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index(typeLdir))
        edge_norms.append([1.])

        src_nodes.append(node_id)
        dst_nodes.append(room_id)
        edge_types.append(rels.index(typeLinv))
        edge_norms.append([1. / max_used_id])

    # self edges
    for node_id in range(n_nodes - 1):
        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index('self'))
        edge_norms.append([1.])

    # Convert outputs to tensors
    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    return src_nodes, dst_nodes, n_nodes, features, None, edge_types, edge_norms, position_by_id, typeMap, []


def initializeAlt2(data):
    # Define variables for edge types and relations
    rels, _ = get_relations('2')
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge

    # Compute data for walls
    Wall = namedtuple('Wall', ['orientation', 'xpos', 'ypos'])
    walls = []
    for wall_index in range(len(data['room']) - 1):
        p1 = np.array(data['room'][wall_index + 0])
        p2 = np.array(data['room'][wall_index + 1])
        dist = np.linalg.norm(p1 - p2)
        iters = int(dist / 400) + 1
        if iters > 1:
            v = (p2 - p1) / iters
            for i in range(iters):
                pa = p1 + v * i
                pb = p1 + v * (i + 1)
                inc2 = pb - pa
                midsp = (pa + pb) / 2
                walls.append(Wall(math.atan2(inc2[0], inc2[1]), midsp[0], midsp[1]))
        else:
            inc = p2 - p1
            midp = (p2 + p1) / 2
            walls.append(Wall(math.atan2(inc[0], inc[1]), midp[0], midp[1]))

    # Compute the number of nodes
    #      room +  room walls      + humans               + objects
    n_nodes = 1 + len(walls) + len(data['humans']) + len(data['objects'])

    # Feature dimensions
    all_features, n_features = get_features('2')
    features = th.zeros(n_nodes, n_features)

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # room (id 0) flobal node
    room_id = 0
    max_used_id = 0
    typeMap[room_id] = 'l'  # 'l' for 'room' (lounge)
    position_by_id[0] = [0, 0]
    features[room_id, all_features.index('room')] = 1.
    features[room_id, all_features.index('room_humans')] = len(data['humans'])
    features[room_id, all_features.index('room_humans2')] = len(data['humans']) * len(data['humans'])

    # humans
    for h in data['humans']:
        src_nodes.append(h['id'])
        dst_nodes.append(room_id)
        edge_types.append(rels.index('p_l'))
        edge_norms.append([1. / len(data['humans'])])

        src_nodes.append(room_id)
        dst_nodes.append(h['id'])
        edge_types.append(rels.index('l_p'))
        edge_norms.append([1.])

        typeMap[h['id']] = 'p'  # 'p' for 'person'
        max_used_id = max(h['id'], max_used_id)
        xpos = float(h['xPos']) / 1000.
        ypos = float(h['yPos']) / 1000.

        position_by_id[h['id']] = [xpos, ypos]
        orientation = float(h['orientation']) / 180. * math.pi
        while orientation > math.pi: orientation -= 2. * math.pi
        while orientation < -math.pi: orientation += 2. * math.pi
        if orientation > math.pi:
            orientation -= math.pi
        elif orientation < -math.pi:
            orientation += math.pi

        features[h['id'], all_features.index('human')] = 1.
        features[h['id'], all_features.index('hum_orientation_sin')] = math.sin(orientation)
        features[h['id'], all_features.index('hum_orientation_cos')] = math.cos(orientation)
        features[h['id'], all_features.index('hum_x_pos')] = 2. * xpos
        features[h['id'], all_features.index('hum_y_pos')] = -2. * ypos

    # objects
    for o in data['objects']:
        src_nodes.append(o['id'])
        dst_nodes.append(room_id)
        edge_types.append(rels.index('o_l'))
        edge_norms.append([1. / len(data['objects'])])

        src_nodes.append(room_id)
        dst_nodes.append(o['id'])
        edge_types.append(rels.index('l_o'))
        edge_norms.append([1.])

        typeMap[o['id']] = 'o'  # 'o' for 'object'
        max_used_id = max(o['id'], max_used_id)
        xpos = float(o['xPos']) / 1000.
        ypos = float(o['yPos']) / 1000.

        position_by_id[o['id']] = [xpos, ypos]
        orientation = float(o['orientation']) / 180. * math.pi
        while orientation > math.pi: orientation -= 2. * math.pi
        while orientation < -math.pi: orientation += 2. * math.pi
        features[o['id'], all_features.index('object')] = 1
        features[o['id'], all_features.index('obj_orientation_sin')] = math.sin(orientation)
        features[o['id'], all_features.index('obj_orientation_cos')] = math.cos(orientation)
        features[o['id'], all_features.index('obj_x_pos')] = 2. * xpos
        features[o['id'], all_features.index('obj_y_pos')] = -2. * ypos

    # walls
    wids = dict()
    for wall in walls:
        max_used_id += 1
        wall_id = max_used_id
        wids[wall] = wall_id
        typeMap[wall_id] = 'w'  # 'w' for 'walls'

        src_nodes.append(wall_id)
        dst_nodes.append(room_id)
        edge_types.append(rels.index('w_l'))
        edge_norms.append([1. / len(walls)])

        src_nodes.append(room_id)
        dst_nodes.append(wall_id)
        edge_types.append(rels.index('l_w'))
        edge_norms.append([1.])

        position_by_id[wall_id] = [wall.xpos / 1000, wall.ypos / 1000]
        features[wall_id, all_features.index('wall')] = 1.
        features[wall_id, all_features.index('wall_orientation_sin')] = math.sin(wall.orientation)
        features[wall_id, all_features.index('wall_orientation_cos')] = math.cos(wall.orientation)
        features[wall_id, all_features.index('wall_x_pos')] = 2. * wall.xpos / 1000.
        features[wall_id, all_features.index('wall_y_pos')] = -2. * wall.ypos / 1000.

    # interactions
    for link in data['links']:
        typeLdir = typeMap[link[0]] + '_' + typeMap[link[1]]
        typeLinv = typeMap[link[1]] + '_' + typeMap[link[0]]

        src_nodes.append(link[0])
        dst_nodes.append(link[1])
        edge_types.append(rels.index(typeLdir))
        edge_norms.append([1.])

        src_nodes.append(link[1])
        dst_nodes.append(link[0])
        edge_types.append(rels.index(typeLinv))
        edge_norms.append([1.])

    # self edges
    for node_id in range(n_nodes):
        r_type = typeMap[node_id].upper()

        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index(r_type))
        edge_norms.append([1.])

    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    return src_nodes, dst_nodes, n_nodes, features, None, edge_types, edge_norms, position_by_id, typeMap, []


def initializeAlt3(data, w_segments=[]):
    # Initialize variables
    rels, num_rels = get_relations('3')
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge
    max_used_id = 0  # Initialise id counter (0 for the robot)

    # Compute data for walls
    Wall = namedtuple('Wall', ['orientation', 'xpos', 'ypos'])
    walls = []
    i_w = 0
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
        else:
            inc = p2 - p1
            midp = (p2 + p1) / 2
            walls.append(Wall(math.atan2(inc[0], inc[1]), midp[0], midp[1]))
        i_w += 1

    # Compute the number of nodes
    # one for the robot + room walls   + humans    + objects              + room(global node)
    n_nodes = 1 + len(walls) + len(data['people']) + len(data['objects']) + 1

    # Feature dimensions
    all_features, n_features = get_features('3')
    features = th.zeros(n_nodes, n_features)
    edge_feats_list = []

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # room (id 0)
    room_id = 0
    typeMap[room_id] = 'r'  # 'r' for 'room'
    position_by_id[room_id] = [0, 0]
    features[room_id, all_features.index('room')] = 1.
    features[room_id, all_features.index('room_humans')] = len(data['people']) / MAX_HUMANS
    features[room_id, all_features.index('room_humans2')] = (len(data['people']) ** 2) / (MAX_HUMANS ** 2)
    features[room_id, all_features.index('robot_adv_vel')] = data['command'][0] / MAX_ADV
    features[room_id, all_features.index('robot_rot_vel')] = data['command'][2] / MAX_ROT
    max_used_id += 1

    # humans
    for h in data['people']:
        src_nodes.append(h['id'])
        dst_nodes.append(room_id)
        edge_types.append(rels.index('p_r'))
        edge_norms.append([1. / len(data['people'])])

        src_nodes.append(room_id)
        dst_nodes.append(h['id'])
        edge_types.append(rels.index('r_p'))
        edge_norms.append([1.])

        typeMap[h['id']] = 'p'  # 'p' for 'person'
        max_used_id += 1
        xpos = h['x'] / 10.
        ypos = h['y'] / 10.
        position_by_id[h['id']] = [xpos, ypos]
        dist = math.sqrt(xpos ** 2 + ypos ** 2)
        va = h['va'] / 10.
        vx = h['vx'] / 10.
        vy = h['vy'] / 10.

        orientation = h['a']
        while orientation > math.pi:
            orientation -= 2. * math.pi
        while orientation < -math.pi:
            orientation += 2. * math.pi
        if orientation > math.pi:
            orientation -= math.pi
        elif orientation < -math.pi:
            orientation += math.pi

        # print(str(math.degrees(angle)) + ' ' + str(math.degrees(orientation)) + ' ' + str(math.degrees(angle_hum)))
        features[h['id'], all_features.index('human')] = 1.
        features[h['id'], all_features.index('hum_orientation_sin')] = math.sin(orientation)
        features[h['id'], all_features.index('hum_orientation_cos')] = math.cos(orientation)
        features[h['id'], all_features.index('hum_x_pos')] = xpos
        features[h['id'], all_features.index('hum_y_pos')] = ypos
        features[h['id'], all_features.index('human_a_vel')] = va
        features[h['id'], all_features.index('human_x_vel')] = vx
        features[h['id'], all_features.index('human_y_vel')] = vy
        features[h['id'], all_features.index('hum_dist')] = dist
        features[h['id'], all_features.index('hum_inv_dist')] = 1. - dist  # /(1.+dist*10.)

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('p_r')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('r_p')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

    # objects
    for o in data['objects']:
        src_nodes.append(o['id'])
        dst_nodes.append(room_id)
        edge_types.append(rels.index('o_r'))
        edge_norms.append([1.])

        src_nodes.append(room_id)
        dst_nodes.append(o['id'])
        edge_types.append(rels.index('r_o'))
        edge_norms.append([1.])

        typeMap[o['id']] = 'o'  # 'o' for 'object'
        max_used_id += 1
        xpos = o['x'] / 10.
        ypos = o['y'] / 10.
        position_by_id[o['id']] = [xpos, ypos]
        dist = math.sqrt(xpos ** 2 + ypos ** 2)
        va = o['va'] / 10.
        vx = o['vx'] / 10.
        vy = o['vy'] / 10.

        orientation = o['a']
        while orientation > math.pi:
            orientation -= 2. * math.pi
        while orientation < -math.pi:
            orientation += 2. * math.pi
        features[o['id'], all_features.index('object')] = 1
        features[o['id'], all_features.index('obj_orientation_sin')] = math.sin(orientation)
        features[o['id'], all_features.index('obj_orientation_cos')] = math.cos(orientation)
        features[o['id'], all_features.index('obj_x_pos')] = xpos
        features[o['id'], all_features.index('obj_y_pos')] = ypos
        features[o['id'], all_features.index('obj_a_vel')] = va
        features[o['id'], all_features.index('obj_x_vel')] = vx
        features[o['id'], all_features.index('obj_y_vel')] = vy
        features[o['id'], all_features.index('obj_x_size')] = o['size_x']
        features[o['id'], all_features.index('obj_y_size')] = o['size_y']
        features[o['id'], all_features.index('obj_dist')] = dist
        features[o['id'], all_features.index('obj_inv_dist')] = 1. - dist  # /(1.+dist*10.)

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('o_r')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('r_o')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

    # Goal
    goal_id = max_used_id
    typeMap[goal_id] = 't'  # 't' for 'goal'
    src_nodes.append(goal_id)
    dst_nodes.append(room_id)
    edge_types.append(rels.index('t_r'))
    edge_norms.append([1.])
    # edge_norms.append([1. / len(data['objects'])])

    src_nodes.append(room_id)
    dst_nodes.append(goal_id)
    edge_types.append(rels.index('r_t'))
    edge_norms.append([1.])

    xpos = data['goal'][0]['x'] / 10.
    ypos = data['goal'][0]['y'] / 10.
    position_by_id[goal_id] = [xpos, ypos]
    dist = math.sqrt(xpos ** 2 + ypos ** 2)
    features[goal_id, all_features.index('goal')] = 1
    features[goal_id, all_features.index('goal_x_pos')] = xpos
    features[goal_id, all_features.index('goal_y_pos')] = ypos
    features[goal_id, all_features.index('goal_dist')] = dist
    features[goal_id, all_features.index('goal_inv_dist')] = 1. - dist  # /(1.+dist*10.)

    max_used_id += 1

    # Edge features
    edge_features = th.zeros(num_rels + 4)
    edge_features[rels.index('t_r')] = 1
    edge_features[-1] = dist
    edge_feats_list.append(edge_features)

    edge_features = th.zeros(num_rels + 4)
    edge_features[rels.index('r_t')] = 1
    edge_features[-1] = dist
    edge_feats_list.append(edge_features)

    # walls
    wids = dict()
    for w_i, wall in enumerate(walls, 0):
        wall_id = max_used_id
        wids[wall] = wall_id
        typeMap[wall_id] = 'w'  # 'w' for 'walls'
        max_used_id += 1

        # # uncomment for links between wall nodes
        # if w_i == len(walls)-1:
        #     next_wall_id = max_used_id-len(walls)
        # else:
        #     next_wall_id = max_used_id
        # # ------------------------------------

        dist = math.sqrt((wall.xpos / 1000.) ** 2 + (wall.ypos / 1000.) ** 2)

        # Links to room node
        src_nodes.append(wall_id)
        dst_nodes.append(room_id)
        edge_types.append(rels.index('w_r'))
        edge_norms.append([1. / len(walls)])

        src_nodes.append(room_id)
        dst_nodes.append(wall_id)
        edge_types.append(rels.index('r_w'))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('w_r')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('r_w')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        # # Links between wall nodes
        # wall_next = walls[(w_i + 1) % len(walls)]
        # dist_wnodes = math.sqrt((wall.xpos / 1000. - wall_next.xpos / 1000.) ** 2 +
        #                         (wall.ypos / 1000. - wall_next.ypos / 1000.) ** 2)

        # src_nodes.append(wall_id)
        # dst_nodes.append(next_wall_id)
        # edge_types.append(rels.index('w_w'))
        # edge_norms.append([1.])

        # src_nodes.append(next_wall_id)
        # dst_nodes.append(wall_id)
        # edge_types.append(rels.index('w_w'))
        # edge_norms.append([1.])

        # edge_features = th.zeros(num_rels + 4)
        # edge_features[rels.index('w_w')] = 1
        # edge_features[-1] = dist_wnodes
        # edge_feats_list.append(edge_features)

        # edge_features = th.zeros(num_rels + 4)
        # edge_features[rels.index('w_w')] = 1
        # edge_features[-1] = dist_wnodes
        # edge_feats_list.append(edge_features)
        # # ----------------------------------

        position_by_id[wall_id] = [wall.xpos / 100., wall.ypos / 100.]

        features[wall_id, all_features.index('wall')] = 1.
        features[wall_id, all_features.index('wall_orientation_sin')] = math.sin(wall.orientation)
        features[wall_id, all_features.index('wall_orientation_cos')] = math.cos(wall.orientation)
        features[wall_id, all_features.index('wall_x_pos')] = wall.xpos / 1000.
        features[wall_id, all_features.index('wall_y_pos')] = wall.ypos / 1000.
        features[wall_id, all_features.index('wall_dist')] = dist
        features[wall_id, all_features.index('wall_inv_dist')] = 1. - dist  # 1./(1.+dist*10.)

    ## Links between walls and people
    # for h in data['people']:
    #     number = 0
    #     for wall in walls:
    #         dist = dist_h_w(h, wall)
    #         if dist < threshold_human_wall:
    #             number -= - 1
    #     for wall in walls:
    #         dist = dist_h_w(h, wall)
    #         if dist < threshold_human_wall:
    #             src_nodes.append(wids[wall])
    #             dst_nodes.append(h['id'])
    #             edge_types.append(rels.index('w_p'))
    #             edge_norms.append([1. / number])

    #             # Edge features
    #             edge_features = th.zeros(num_rels + 4)
    #             edge_features[rels.index('w_p')] = 1
    #             edge_features[-1] = dist
    #             edge_feats_list.append(edge_features)

    # for wall in walls:
    #     number = 0
    #     for h in data['people']:
    #         dist = dist_h_w(h, wall)
    #         if dist < threshold_human_wall:
    #             number -= - 1
    #     for h in data['people']:
    #         dist = dist_h_w(h, wall)
    #         if dist < threshold_human_wall:
    #             src_nodes.append(h['id'])
    #             dst_nodes.append(wids[wall])
    #             edge_types.append(rels.index('p_w'))
    #             edge_norms.append([1. / number])

    #             # Edge features
    #             edge_features = th.zeros(num_rels + 4)
    #             edge_features[rels.index('p_w')] = 1
    #             edge_features[-1] = dist
    #             edge_feats_list.append(edge_features)
    # # ----------------------------------

    # interaction links
    for link in data['interaction']:
        typeLdir = typeMap[link['src']] + '_' + typeMap[link['dst']]
        typeLinv = typeMap[link['dst']] + '_' + typeMap[link['src']]

        dist = math.sqrt((position_by_id[link['src']][0] - position_by_id[link['dst']][0]) ** 2 +
                         (position_by_id[link['src']][1] - position_by_id[link['dst']][1]) ** 2)

        src_nodes.append(link['src'])
        dst_nodes.append(link['dst'])
        edge_types.append(rels.index(typeLdir))
        edge_norms.append([1.])

        src_nodes.append(link['dst'])
        dst_nodes.append(link['src'])
        edge_types.append(rels.index(typeLinv))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index(typeLdir)] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index(typeLinv)] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

    # self edges
    for node_id in range(n_nodes):
        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index('self'))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('self')] = 1
        edge_features[-1] = 0
        edge_feats_list.append(edge_features)

    # Convert outputs to tensors
    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    edge_feats = th.stack(edge_feats_list)

    return src_nodes, dst_nodes, n_nodes, features, edge_feats, edge_types, edge_norms, position_by_id, typeMap, \
           w_segments


def initializeAlt4(data, w_segments=[]):
    # Initialize variables
    rels, num_rels = get_relations('4')
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge
    max_used_id = 0  # Initialise id counter (0 for the robot)

    # Compute the number of nodes
    # one for the robot  + objects  + one for the goal
    n_nodes = 1 + len(data['objects']) + 1

    # Feature dimensions
    all_features, n_features = get_features('4')
    features = th.zeros(n_nodes, n_features)
    edge_feats_list = []

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # room (id 0)
    room_id = 0
    typeMap[room_id] = 'r'  # 'r' for 'room'
    position_by_id[room_id] = [0, 0]
    features[room_id, all_features.index('room')] = 1.
    features[room_id, all_features.index('robot_adv_vel')] = data['command'][0] / MAX_ADV
    features[room_id, all_features.index('robot_rot_vel')] = data['command'][2] / MAX_ROT
    max_used_id += 1

    # objects
    for o in data['objects']:
        src_nodes.append(o['id'])
        dst_nodes.append(room_id)
        edge_types.append(rels.index('o_r'))
        edge_norms.append([1.])

        src_nodes.append(room_id)
        dst_nodes.append(o['id'])
        edge_types.append(rels.index('r_o'))
        edge_norms.append([1.])

        typeMap[o['id']] = 'o'  # 'o' for 'object'
        max_used_id += 1
        xpos = o['x'] / 10.
        ypos = o['y'] / 10.
        position_by_id[o['id']] = [xpos, ypos]
        dist = math.sqrt(xpos ** 2 + ypos ** 2)
        va = o['va'] / 10.
        vx = o['vx'] / 10.
        vy = o['vy'] / 10.

        orientation = o['a']
        while orientation > math.pi:
            orientation -= 2. * math.pi
        while orientation < -math.pi:
            orientation += 2. * math.pi
        features[o['id'], all_features.index('object')] = 1
        features[o['id'], all_features.index('obj_orientation_sin')] = math.sin(orientation)
        features[o['id'], all_features.index('obj_orientation_cos')] = math.cos(orientation)
        features[o['id'], all_features.index('obj_x_pos')] = xpos
        features[o['id'], all_features.index('obj_y_pos')] = ypos
        features[o['id'], all_features.index('obj_a_vel')] = va
        features[o['id'], all_features.index('obj_x_vel')] = vx
        features[o['id'], all_features.index('obj_y_vel')] = vy
        features[o['id'], all_features.index('obj_x_size')] = o['size_x']
        features[o['id'], all_features.index('obj_y_size')] = o['size_y']
        features[o['id'], all_features.index('obj_dist')] = dist
        features[o['id'], all_features.index('obj_inv_dist')] = 1. - dist  # /(1.+dist*10.)

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('o_r')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('r_o')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

    # Goal
    goal_id = max_used_id
    typeMap[goal_id] = 'g'  # 'g' for 'goal'
    src_nodes.append(goal_id)
    dst_nodes.append(room_id)
    edge_types.append(rels.index('g_r'))
    edge_norms.append([1.])
    # edge_norms.append([1. / len(data['objects'])])

    src_nodes.append(room_id)
    dst_nodes.append(goal_id)
    edge_types.append(rels.index('r_g'))
    edge_norms.append([1.])

    xpos = data['goal'][0]['x'] / 10.
    ypos = data['goal'][0]['y'] / 10.
    position_by_id[goal_id] = [xpos, ypos]
    dist = math.sqrt(xpos ** 2 + ypos ** 2)
    features[goal_id, all_features.index('goal')] = 1
    features[goal_id, all_features.index('goal_x_pos')] = xpos
    features[goal_id, all_features.index('goal_y_pos')] = ypos
    features[goal_id, all_features.index('goal_dist')] = dist
    features[goal_id, all_features.index('goal_inv_dist')] = 1. - dist  # /(1.+dist*10.)

    max_used_id += 1

    # Edge features
    edge_features = th.zeros(num_rels + 4)
    edge_features[rels.index('g_r')] = 1
    edge_features[-1] = dist
    edge_feats_list.append(edge_features)

    edge_features = th.zeros(num_rels + 4)
    edge_features[rels.index('r_g')] = 1
    edge_features[-1] = dist
    edge_feats_list.append(edge_features)

    # self edges
    for node_id in range(n_nodes):
        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index('self'))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('self')] = 1
        edge_features[-1] = 0
        edge_feats_list.append(edge_features)

    # Convert outputs to tensors
    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    edge_feats = th.stack(edge_feats_list)

    return src_nodes, dst_nodes, n_nodes, features, edge_feats, edge_types, edge_norms, position_by_id, typeMap, \
           []


def initializeAlt5(data, w_segments=[]):
    # Initialize variables
    rels, num_rels = get_relations('5')
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge
    max_used_id = 0  # Initialise id counter (0 for the robot)

    # Compute the number of nodes
    # one for the robot  + humans  + one for the goal
    n_nodes = 1 + len(data['people']) + 1

    # Feature dimensions
    all_features, n_features = get_features('5')
    features = th.zeros(n_nodes, n_features)
    edge_feats_list = []

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # room (id 0)
    room_id = 0
    typeMap[room_id] = 'r'  # 'r' for 'room'
    position_by_id[room_id] = [0, 0]
    features[room_id, all_features.index('room')] = 1.
    features[room_id, all_features.index('robot_adv_vel')] = data['command'][0] / MAX_ADV
    features[room_id, all_features.index('robot_rot_vel')] = data['command'][2] / MAX_ROT
    max_used_id += 1

    # humans
    for h in data['people']:
        h_id = max_used_id
        src_nodes.append(h_id)
        dst_nodes.append(room_id)
        edge_types.append(rels.index('p_r'))
        edge_norms.append([1. / len(data['people'])])

        src_nodes.append(room_id)
        dst_nodes.append(h_id)
        edge_types.append(rels.index('r_p'))
        edge_norms.append([1.])

        typeMap[h_id] = 'p'  # 'p' for 'person'
        xpos = h['x'] / 10.
        ypos = h['y'] / 10.
        position_by_id[h_id] = [xpos, ypos]
        dist = math.sqrt(xpos ** 2 + ypos ** 2)
        va = h['va'] / 10.
        vx = h['vx'] / 10.
        vy = h['vy'] / 10.

        max_used_id += 1
        orientation = h['a']

        # print(str(math.degrees(angle)) + ' ' + str(math.degrees(orientation)) + ' ' + str(math.degrees(angle_hum)))
        features[h_id, all_features.index('human')] = 1.
        features[h_id, all_features.index('hum_orientation_sin')] = math.sin(orientation)
        features[h_id, all_features.index('hum_orientation_cos')] = math.cos(orientation)
        features[h_id, all_features.index('hum_x_pos')] = xpos
        features[h_id, all_features.index('hum_y_pos')] = ypos
        features[h_id, all_features.index('human_a_vel')] = va
        features[h_id, all_features.index('human_x_vel')] = vx
        features[h_id, all_features.index('human_y_vel')] = vy
        features[h_id, all_features.index('hum_dist')] = dist
        features[h_id, all_features.index('hum_inv_dist')] = 1. - dist  # /(1.+dist*10.)

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('p_r')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('r_p')] = 1
        edge_features[-1] = dist
        edge_feats_list.append(edge_features)

    # Goal
    goal_id = max_used_id
    typeMap[goal_id] = 'g'  # 'g' for 'goal'
    src_nodes.append(goal_id)
    dst_nodes.append(room_id)
    edge_types.append(rels.index('g_r'))
    edge_norms.append([1.])
    # edge_norms.append([1. / len(data['objects'])])

    src_nodes.append(room_id)
    dst_nodes.append(goal_id)
    edge_types.append(rels.index('r_g'))
    edge_norms.append([1.])

    xpos = data['goal'][0]['x'] / 10.
    ypos = data['goal'][0]['y'] / 10.
    position_by_id[goal_id] = [xpos, ypos]
    dist = math.sqrt(xpos ** 2 + ypos ** 2)
    features[goal_id, all_features.index('goal')] = 1
    features[goal_id, all_features.index('goal_x_pos')] = xpos
    features[goal_id, all_features.index('goal_y_pos')] = ypos
    features[goal_id, all_features.index('goal_dist')] = dist
    features[goal_id, all_features.index('goal_inv_dist')] = 1. - dist  # /(1.+dist*10.)

    max_used_id += 1

    # Edge features
    edge_features = th.zeros(num_rels + 4)
    edge_features[rels.index('g_r')] = 1
    edge_features[-1] = dist
    edge_feats_list.append(edge_features)

    edge_features = th.zeros(num_rels + 4)
    edge_features[rels.index('r_g')] = 1
    edge_features[-1] = dist
    edge_feats_list.append(edge_features)

    # self edges
    for node_id in range(n_nodes):
        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index('self'))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('self')] = 1
        edge_features[-1] = 0
        edge_feats_list.append(edge_features)

    # Convert outputs to tensors
    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    edge_feats = th.stack(edge_feats_list)

    return src_nodes, dst_nodes, n_nodes, features, edge_feats, edge_types, edge_norms, position_by_id, typeMap, \
           []


# Initialize alternatives 6 and 7:
# 6: people, goal and time features
# 7: people, walls, goal and time features. Can be combined with the grid
def initializeAlt6(data_sequence, alt='6', w_segments=[]):
    # Initialize variables
    rels, num_rels = get_relations(alt)
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge

    # Compute the number of nodes
    # one for the robot  + humans  + one for the goal
    n_nodes = 1 + len(data_sequence[0]['people']) + 1

    if alt == '7':
        walls, w_segments, _ = generate_walls_information(data_sequence[0], w_segments)
        n_nodes += len(walls)

    # Feature dimensions
    all_features, n_features = get_features(alt)
    # print(all_features, n_features)
    features = th.zeros(n_nodes, n_features)
    edge_feats_list = []

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    t_tag = ['']
    for i in range(1, N_INTERVALS):
        t_tag.append('_t' + str(i))

    n_instants = 0
    frames_in_interval = []
    first_frame = True
    for data in data_sequence:
        if n_instants == N_INTERVALS:
            break
        if not first_frame and math.fabs(
                data['timestamp'] - frames_in_interval[-1]['timestamp']) < FRAMES_INTERVAL:  # Truncated to N seconds
            continue

        frames_in_interval.append(data)

        max_used_id = 0  # Initialise id counter (0 for the robot)
        # room (id 0)
        room_id = 0

        if first_frame:
            typeMap[room_id] = 'r'  # 'r' for 'room'
            position_by_id[room_id] = [0, 0]
            features[room_id, all_features.index('room')] = 1.

        features[room_id, all_features.index('step_fraction' + t_tag[n_instants])] = data['step_fraction']
        features[room_id, all_features.index('robot_adv_vel' + t_tag[n_instants])] = data['command'][0] / MAX_ADV
        features[room_id, all_features.index('robot_rot_vel' + t_tag[n_instants])] = data['command'][2] / MAX_ROT
        features[room_id, all_features.index('t' + str(n_instants))] = 1.

        max_used_id += 1

        # humans
        for h in data['people']:
            h_id = max_used_id

            xpos = h['x'] / 10.
            ypos = h['y'] / 10.
            dist = math.sqrt(xpos ** 2 + ypos ** 2)
            va = h['va'] / 10.
            vx = h['vx'] / 10.
            vy = h['vy'] / 10.
            orientation = h['a']

            if first_frame:
                src_nodes.append(h_id)
                dst_nodes.append(room_id)
                edge_types.append(rels.index('p_r'))
                edge_norms.append([1. / len(data['people'])])

                src_nodes.append(room_id)
                dst_nodes.append(h_id)
                edge_types.append(rels.index('r_p'))
                edge_norms.append([1.])

                # Edge features
                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('p_r')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)

                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('r_p')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)

                typeMap[h_id] = 'p'  # 'p' for 'person'
                position_by_id[h_id] = [xpos, ypos]
                features[h_id, all_features.index('human')] = 1.

            max_used_id += 1

            features[h_id, all_features.index('step_fraction' + t_tag[n_instants])] = data['step_fraction']
            features[h_id, all_features.index('hum_orientation_sin' + t_tag[n_instants])] = math.sin(orientation)
            features[h_id, all_features.index('hum_orientation_cos' + t_tag[n_instants])] = math.cos(orientation)
            features[h_id, all_features.index('hum_x_pos' + t_tag[n_instants])] = xpos
            features[h_id, all_features.index('hum_y_pos' + t_tag[n_instants])] = ypos
            features[h_id, all_features.index('human_a_vel' + t_tag[n_instants])] = va
            features[h_id, all_features.index('human_x_vel' + t_tag[n_instants])] = vx
            features[h_id, all_features.index('human_y_vel' + t_tag[n_instants])] = vy
            features[h_id, all_features.index('hum_dist' + t_tag[n_instants])] = dist
            features[h_id, all_features.index('hum_inv_dist' + t_tag[n_instants])] = 1. - dist  # /(1.+dist*10.)
            features[h_id, all_features.index('t' + str(n_instants))] = 1.

        # Goal
        goal_id = max_used_id
        max_used_id += 1

        xpos = data['goal'][0]['x'] / 10.
        ypos = data['goal'][0]['y'] / 10.
        dist = math.sqrt(xpos ** 2 + ypos ** 2)

        if first_frame:
            typeMap[goal_id] = 't'  # 't' for 'goal'
            src_nodes.append(goal_id)
            dst_nodes.append(room_id)
            edge_types.append(rels.index('t_r'))
            edge_norms.append([1.])
            # edge_norms.append([1. / len(data['objects'])])

            src_nodes.append(room_id)
            dst_nodes.append(goal_id)
            edge_types.append(rels.index('r_t'))
            edge_norms.append([1.])

            # Edge features
            edge_features = th.zeros(num_rels + 4)
            edge_features[rels.index('t_r')] = 1
            edge_features[-1] = dist
            edge_feats_list.append(edge_features)

            edge_features = th.zeros(num_rels + 4)
            edge_features[rels.index('r_t')] = 1
            edge_features[-1] = dist
            edge_feats_list.append(edge_features)

            position_by_id[goal_id] = [xpos, ypos]

            features[goal_id, all_features.index('goal')] = 1

        features[goal_id, all_features.index('step_fraction' + t_tag[n_instants])] = data['step_fraction']
        features[goal_id, all_features.index('goal_x_pos' + t_tag[n_instants])] = xpos
        features[goal_id, all_features.index('goal_y_pos' + t_tag[n_instants])] = ypos
        features[goal_id, all_features.index('goal_dist' + t_tag[n_instants])] = dist
        features[goal_id, all_features.index('goal_inv_dist' + t_tag[n_instants])] = 1. - dist  # /(1.+dist*10.)
        features[goal_id, all_features.index('t' + str(n_instants))] = 1.

        # Walls
        if alt == '7':
            if not first_frame:
                walls, w_segments, _ = generate_walls_information(data, w_segments)

            for wall in walls:
                wall_id = max_used_id
                max_used_id += 1

                if first_frame:
                    typeMap[wall_id] = 'w'  # 'w' for 'walls'

                    dist = math.sqrt((wall.xpos / 1000.) ** 2 + (wall.ypos / 1000.) ** 2)

                    # Links to room node
                    src_nodes.append(wall_id)
                    dst_nodes.append(room_id)
                    edge_types.append(rels.index('w_r'))
                    edge_norms.append([1. / len(walls)])

                    src_nodes.append(room_id)
                    dst_nodes.append(wall_id)
                    edge_types.append(rels.index('r_w'))
                    edge_norms.append([1.])

                    # Edge features
                    edge_features = th.zeros(num_rels + 4)
                    edge_features[rels.index('w_r')] = 1
                    edge_features[-1] = dist
                    edge_feats_list.append(edge_features)

                    edge_features = th.zeros(num_rels + 4)
                    edge_features[rels.index('r_w')] = 1
                    edge_features[-1] = dist
                    edge_feats_list.append(edge_features)

                    position_by_id[wall_id] = [wall.xpos / 1000., wall.ypos / 1000.]
                    features[wall_id, all_features.index('wall')] = 1.

                features[wall_id, all_features.index('step_fraction' + t_tag[n_instants])] = data['step_fraction']
                features[wall_id, all_features.index('wall_orientation_sin' + t_tag[n_instants])] = math.sin(
                    wall.orientation)
                features[wall_id, all_features.index('wall_orientation_cos' + t_tag[n_instants])] = math.cos(
                    wall.orientation)
                features[wall_id, all_features.index('wall_x_pos' + t_tag[n_instants])] = wall.xpos / 1000.
                features[wall_id, all_features.index('wall_y_pos' + t_tag[n_instants])] = wall.ypos / 1000.
                features[wall_id, all_features.index('wall_dist' + t_tag[n_instants])] = dist
                features[
                    wall_id, all_features.index('wall_inv_dist' + t_tag[n_instants])] = 1. - dist  # 1./(1.+dist*10.)

        n_instants += 1
        first_frame = False

    # self edges
    for node_id in range(n_nodes):
        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index('self'))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(num_rels + 4)
        edge_features[rels.index('self')] = 1
        edge_features[-1] = 0
        edge_feats_list.append(edge_features)

    # Convert outputs to tensors
    src_nodes = th.IntTensor(src_nodes)
    dst_nodes = th.IntTensor(dst_nodes)

    edge_types = th.IntTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    edge_feats = th.stack(edge_feats_list)

    return src_nodes, dst_nodes, n_nodes, features, edge_feats, edge_types, edge_norms, position_by_id, typeMap, \
           []


# Initialize alternative 8: people, objects, goal, walls, interactions and time features. Can be combined with the grid
def initializeAlt8(data_sequence, alt='8', wall_segments=None, with_edge_features=False):
    # Initialize variables
    if wall_segments is None:
        wall_segments = []
    rels, num_rels = get_relations(alt)
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge

    # Compute the number of nodes
    # one for the robot  + humans  + one for the goal
    n_nodes = 1 + len(data_sequence[0]['people']) + len(data_sequence[0]['objects']) + 1

    walls, wall_segments, _ = generate_walls_information(data_sequence[0], wall_segments)
    n_nodes += len(walls)

    # Feature dimensions
    all_features, n_features = get_features(alt)
    # print(all_features, n_features)
    features = th.zeros(n_nodes, n_features)
    edge_feats_list = []

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    t_tag = ['']
    for i in range(1, N_INTERVALS):
        t_tag.append('_t' + str(i))

    if 'step_fraction' in data_sequence[0].keys():
        step_fraction = data_sequence[0]['step_fraction']
    else:
        step_fraction = 0

    n_instants = 0
    frames_in_interval = []
    first_frame = True
    for data in data_sequence:
        if n_instants == N_INTERVALS:
            break
        if not first_frame and math.fabs(
                data['timestamp'] - frames_in_interval[-1]['timestamp']) < FRAMES_INTERVAL:  # Truncated to N seconds
            continue

        if 'step_fraction' in data.keys():
            step_fraction = data['step_fraction']
        else:
            step_fraction = 0

        frames_in_interval.append(data)

        max_used_id = 0  # Initialise id counter (0 for the robot)
        # room (id 0)
        room_id = 0

        if first_frame:
            typeMap[room_id] = 'r'  # 'r' for 'room'
            position_by_id[room_id] = [0, 0]
            features[room_id, all_features.index('room')] = 1.

        features[room_id, all_features.index('step_fraction' + t_tag[n_instants])] = step_fraction
        features[room_id, all_features.index('robot_adv_vel' + t_tag[n_instants])] = data['command'][0] / MAX_ADV
        features[room_id, all_features.index('robot_rot_vel' + t_tag[n_instants])] = data['command'][2] / MAX_ROT
        features[room_id, all_features.index('t' + str(n_instants))] = 1.

        max_used_id += 1

        # objects
        for o in data['objects']:
            o_id = o['id']

            xpos = o['x'] / 10.
            ypos = o['y'] / 10.

            dist = math.sqrt(xpos ** 2 + ypos ** 2)
            va = o['va'] / 10.
            vx = o['vx'] / 10.
            vy = o['vy'] / 10.
            orientation = o['a']

            if first_frame:
                src_nodes.append(o_id)
                dst_nodes.append(room_id)
                edge_types.append(rels.index('o_r'))
                edge_norms.append([1.])

                src_nodes.append(room_id)
                dst_nodes.append(o_id)
                edge_types.append(rels.index('r_o'))
                edge_norms.append([1.])
                # Edge features
                if with_edge_features:
                    edge_features = th.zeros(num_rels + 4)
                    edge_features[rels.index('o_r')] = 1
                    edge_features[-1] = dist
                    edge_feats_list.append(edge_features)

                    edge_features = th.zeros(num_rels + 4)
                    edge_features[rels.index('r_o')] = 1
                    edge_features[-1] = dist
                    edge_feats_list.append(edge_features)

                typeMap[o_id] = 'o'  # 'o' for 'object'
                position_by_id[o_id] = [xpos, ypos]
                features[o_id, all_features.index('object')] = 1

            max_used_id += 1

            features[o_id, all_features.index('step_fraction' + t_tag[n_instants])] = step_fraction
            features[o_id, all_features.index('obj_orientation_sin' + t_tag[n_instants])] = math.sin(orientation)
            features[o_id, all_features.index('obj_orientation_cos' + t_tag[n_instants])] = math.cos(orientation)
            features[o_id, all_features.index('obj_x_pos' + t_tag[n_instants])] = xpos
            features[o_id, all_features.index('obj_y_pos' + t_tag[n_instants])] = ypos
            features[o_id, all_features.index('obj_a_vel' + t_tag[n_instants])] = va
            features[o_id, all_features.index('obj_x_vel' + t_tag[n_instants])] = vx
            features[o_id, all_features.index('obj_y_vel' + t_tag[n_instants])] = vy
            features[o_id, all_features.index('obj_x_size' + t_tag[n_instants])] = o['size_x']
            features[o_id, all_features.index('obj_y_size' + t_tag[n_instants])] = o['size_y']
            features[o_id, all_features.index('obj_dist' + t_tag[n_instants])] = dist
            features[o_id, all_features.index('obj_inv_dist' + t_tag[n_instants])] = 1. - dist  # /(1.+dist*10.)

        # humans
        for h in data['people']:
            h_id = h['id']

            xpos = h['x'] / 10.
            ypos = h['y'] / 10.
            dist = math.sqrt(xpos ** 2 + ypos ** 2)
            va = h['va'] / 10.
            vx = h['vx'] / 10.
            vy = h['vy'] / 10.
            orientation = h['a']

            if first_frame:
                src_nodes.append(h_id)
                dst_nodes.append(room_id)
                edge_types.append(rels.index('p_r'))
                edge_norms.append([1. / len(data['people'])])

                src_nodes.append(room_id)
                dst_nodes.append(h_id)
                edge_types.append(rels.index('r_p'))
                edge_norms.append([1.])

                # Edge features
                if with_edge_features:
                    edge_features = th.zeros(num_rels + 4)
                    edge_features[rels.index('p_r')] = 1
                    edge_features[-1] = dist
                    edge_feats_list.append(edge_features)

                    edge_features = th.zeros(num_rels + 4)
                    edge_features[rels.index('r_p')] = 1
                    edge_features[-1] = dist
                    edge_feats_list.append(edge_features)

                typeMap[h_id] = 'p'  # 'p' for 'person'
                position_by_id[h_id] = [xpos, ypos]
                features[h_id, all_features.index('human')] = 1.

            max_used_id += 1

            features[h_id, all_features.index('step_fraction' + t_tag[n_instants])] = step_fraction
            features[h_id, all_features.index('hum_orientation_sin' + t_tag[n_instants])] = math.sin(orientation)
            features[h_id, all_features.index('hum_orientation_cos' + t_tag[n_instants])] = math.cos(orientation)
            features[h_id, all_features.index('hum_x_pos' + t_tag[n_instants])] = xpos
            features[h_id, all_features.index('hum_y_pos' + t_tag[n_instants])] = ypos
            features[h_id, all_features.index('human_a_vel' + t_tag[n_instants])] = va
            features[h_id, all_features.index('human_x_vel' + t_tag[n_instants])] = vx
            features[h_id, all_features.index('human_y_vel' + t_tag[n_instants])] = vy
            features[h_id, all_features.index('hum_dist' + t_tag[n_instants])] = dist
            features[h_id, all_features.index('hum_inv_dist' + t_tag[n_instants])] = 1. - dist  # /(1.+dist*10.)
            features[h_id, all_features.index('t' + str(n_instants))] = 1.

        # Goal
        goal_id = max_used_id
        max_used_id += 1

        xpos = data['goal'][0]['x'] / 10.
        ypos = data['goal'][0]['y'] / 10.
        dist = math.sqrt(xpos ** 2 + ypos ** 2)

        if first_frame:
            typeMap[goal_id] = 't'  # 't' for 'goal'
            src_nodes.append(goal_id)
            dst_nodes.append(room_id)
            edge_types.append(rels.index('t_r'))
            edge_norms.append([1.])
            # edge_norms.append([1. / len(data['objects'])])

            src_nodes.append(room_id)
            dst_nodes.append(goal_id)
            edge_types.append(rels.index('r_t'))
            edge_norms.append([1.])

            # Edge features
            if with_edge_features:
                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('t_r')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)

                edge_features = th.zeros(num_rels + 4)
                edge_features[rels.index('r_t')] = 1
                edge_features[-1] = dist
                edge_feats_list.append(edge_features)

            position_by_id[goal_id] = [xpos, ypos]

            features[goal_id, all_features.index('goal')] = 1

        features[goal_id, all_features.index('step_fraction' + t_tag[n_instants])] = step_fraction
        features[goal_id, all_features.index('goal_x_pos' + t_tag[n_instants])] = xpos
        features[goal_id, all_features.index('goal_y_pos' + t_tag[n_instants])] = ypos
        features[goal_id, all_features.index('goal_dist' + t_tag[n_instants])] = dist
        features[goal_id, all_features.index('goal_inv_dist' + t_tag[n_instants])] = 1. - dist  # /(1.+dist*10.)
        features[goal_id, all_features.index('t' + str(n_instants))] = 1.

        # Walls
        if not first_frame:
            walls, wall_segments, _ = generate_walls_information(data, wall_segments)

        for wall in walls:
            wall_id = max_used_id
            max_used_id += 1

            if first_frame:
                typeMap[wall_id] = 'w'  # 'w' for 'walls'

                dist = math.sqrt((wall.xpos / 1000.) ** 2 + (wall.ypos / 1000.) ** 2)

                # Links to room node
                src_nodes.append(wall_id)
                dst_nodes.append(room_id)
                edge_types.append(rels.index('w_r'))
                edge_norms.append([1. / len(walls)])

                src_nodes.append(room_id)
                dst_nodes.append(wall_id)
                edge_types.append(rels.index('r_w'))
                edge_norms.append([1.])

                # Edge features
                if with_edge_features:
                    edge_features = th.zeros(num_rels + 4)
                    edge_features[rels.index('w_r')] = 1
                    edge_features[-1] = dist
                    edge_feats_list.append(edge_features)

                    edge_features = th.zeros(num_rels + 4)
                    edge_features[rels.index('r_w')] = 1
                    edge_features[-1] = dist
                    edge_feats_list.append(edge_features)

                position_by_id[wall_id] = [wall.xpos / 1000., wall.ypos / 1000.]
                features[wall_id, all_features.index('wall')] = 1.

            features[wall_id, all_features.index('step_fraction' + t_tag[n_instants])] = step_fraction
            features[wall_id, all_features.index('wall_orientation_sin' + t_tag[n_instants])] = math.sin(
                wall.orientation)
            features[wall_id, all_features.index('wall_orientation_cos' + t_tag[n_instants])] = math.cos(
                wall.orientation)
            features[wall_id, all_features.index('wall_x_pos' + t_tag[n_instants])] = wall.xpos / 1000.
            features[wall_id, all_features.index('wall_y_pos' + t_tag[n_instants])] = wall.ypos / 1000.
            features[wall_id, all_features.index('wall_dist' + t_tag[n_instants])] = dist
            features[wall_id, all_features.index('wall_inv_dist' + t_tag[n_instants])] = 1. - dist  # 1./(1.+dist*10.)
            features[wall_id, all_features.index('t' + str(n_instants))] = 1.

        n_instants += 1
        first_frame = False

    # interaction links
    for link in data['interaction']:
        typeLdir = typeMap[link['src']] + '_' + typeMap[link['dst']]
        typeLinv = typeMap[link['dst']] + '_' + typeMap[link['src']]

        dist = math.sqrt((position_by_id[link['src']][0] - position_by_id[link['dst']][0]) ** 2 +
                         (position_by_id[link['src']][1] - position_by_id[link['dst']][1]) ** 2)

        # print(link)
        src_nodes.append(link['src'])
        dst_nodes.append(link['dst'])
        edge_types.append(rels.index(typeLdir))
        edge_norms.append([1.])

        src_nodes.append(link['dst'])
        dst_nodes.append(link['src'])
        edge_types.append(rels.index(typeLinv))
        edge_norms.append([1.])

        # Edge features
        if with_edge_features:
            edge_features = th.zeros(num_rels + 4)
            edge_features[rels.index(typeLdir)] = 1
            edge_features[-1] = dist
            edge_feats_list.append(edge_features)

            edge_features = th.zeros(num_rels + 4)
            edge_features[rels.index(typeLinv)] = 1
            edge_features[-1] = dist
            edge_feats_list.append(edge_features)

    # self edges
    for node_id in range(n_nodes):
        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index('self'))
        edge_norms.append([1.])

        # Edge features
        if with_edge_features:
            edge_features = th.zeros(num_rels + 4)
            edge_features[rels.index('self')] = 1
            edge_features[-1] = 0
            edge_feats_list.append(edge_features)

    # Convert outputs to tensors
    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    if with_edge_features:
        edge_feats = th.stack(edge_feats_list)
    else:
        edge_feats = th.empty(0,0)

    return src_nodes, dst_nodes, n_nodes, features, edge_feats, edge_types, edge_norms, position_by_id, typeMap, \
           []


def initializeAlt9(data_sequence, alt='9', w_segments=[]):
    # Initialize variables
    rels, num_rels = get_relations(alt)
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge

    # Compute the number of nodes
    # one for the robot  + humans  + one for the goal
    n_nodes = 1 + len(data_sequence[0]['people']) + len(data_sequence[0]['objects']) + 1

    walls, w_segments, wall_index = generate_walls_information(data_sequence[0], w_segments)
    n_nodes += len(walls)

    # Feature dimensions
    all_features, n_features = get_features(alt)
    all_edge_features, n_edge_features = get_edge_features(alt)
    # print(all_features, n_features)
    edge_features_1 = th.zeros(n_nodes, n_edge_features) #[]
    edge_features_2 = th.zeros(n_nodes, n_edge_features) #[]
    features = th.zeros(n_nodes, n_features)
    edge_feats_list = []

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    t_tag = ['']
    for i in range(1, N_INTERVALS):
        t_tag.append('_t' + str(i))

    n_instants = 0
    frames_in_interval = []
    first_frame = True
    last_frame = False
    for data in data_sequence:
        if n_instants == N_INTERVALS:
            break
        if not first_frame and math.fabs(
                data['timestamp'] - frames_in_interval[-1]['timestamp']) < FRAMES_INTERVAL:  # Truncated to N seconds
            continue

        if n_instants >= N_INTERVALS-1:
            last_frame = True

        if 'step_fraction' in data.keys():
            step_fraction = data['step_fraction']
        else:
            step_fraction = 0

        frames_in_interval.append(data)

        max_used_id = 0  # Initialise id counter (0 for the robot)
        # room (id 0)
        room_id = 0

        if first_frame:
            typeMap[room_id] = 'r'  # 'r' for 'room'
            position_by_id[room_id] = [0, 0, 0]
            features[room_id, all_features.index('room')] = 1.

        features[room_id, all_features.index('step_fraction' + t_tag[n_instants])] = step_fraction
        features[room_id, all_features.index('robot_adv_vel' + t_tag[n_instants])] = data['command'][0] / MAX_ADV
        features[room_id, all_features.index('robot_rot_vel' + t_tag[n_instants])] = data['command'][2] / MAX_ROT
        features[room_id, all_features.index('t' + str(n_instants))] = 1.

        max_used_id += 1

        # objects
        for o in data['objects']:
            o_id = o['id']

            xpos = o['x'] / 10.
            ypos = o['y'] / 10.

            dist = math.sqrt(xpos ** 2 + ypos ** 2)
            va = o['va'] / 10.
            vx = o['vx'] / 10.
            vy = o['vy'] / 10.
            orientation = o['a']

            if first_frame:
                typeMap[o_id] = 'o'  # 'o' for 'object'
                position_by_id[o_id] = [xpos, ypos, orientation]
                features[o_id, all_features.index('object')] = 1

                src_nodes.append(o_id)
                dst_nodes.append(room_id)
                edge_types.append(rels.index('o_r'))
                edge_norms.append([1.])

                src_nodes.append(room_id)
                dst_nodes.append(o_id)
                edge_types.append(rels.index('r_o'))
                edge_norms.append([1.])

                # Edge features
                edge_features_1[o_id] = th.zeros(n_edge_features)
                edge_features_1[o_id][all_edge_features.index('o_r')] = 1.
                edge_features_2[o_id] = th.zeros(n_edge_features)
                edge_features_2[o_id][all_edge_features.index('r_o')] = 1.

            # Edge features
            t_collision = math.tanh(o['t_collision']/3)
            rx, ry, ra = calculate_relative_position(entity1=(xpos, ypos, orientation), entity2=(0, 0, 0))
            edge_features_1[o_id][all_edge_features.index('x' + t_tag[n_instants])] = rx
            edge_features_1[o_id][all_edge_features.index('y' + t_tag[n_instants])] = ry
            edge_features_1[o_id][all_edge_features.index('orientation' + t_tag[n_instants])] = ra
            edge_features_1[o_id][all_edge_features.index('t_collision' + t_tag[n_instants])] = t_collision

            rx, ry, ra = calculate_relative_position(entity1=(0, 0, 0), entity2=(xpos, ypos, orientation))
            edge_features_2[o_id][all_edge_features.index('x' + t_tag[n_instants])] = rx
            edge_features_2[o_id][all_edge_features.index('y' + t_tag[n_instants])] = ry
            edge_features_2[o_id][all_edge_features.index('orientation' + t_tag[n_instants])] = ra
            edge_features_2[o_id][all_edge_features.index('t_collision' + t_tag[n_instants])] = t_collision

            if last_frame:
                edge_feats_list.append(edge_features_1[o_id])
                edge_feats_list.append(edge_features_2[o_id])

            # Node features
            features[o_id, all_features.index('step_fraction' + t_tag[n_instants])] = step_fraction
            features[o_id, all_features.index('obj_orientation_sin' + t_tag[n_instants])] = math.sin(orientation)
            features[o_id, all_features.index('obj_orientation_cos' + t_tag[n_instants])] = math.cos(orientation)
            features[o_id, all_features.index('obj_x_pos' + t_tag[n_instants])] = xpos
            features[o_id, all_features.index('obj_y_pos' + t_tag[n_instants])] = ypos
            features[o_id, all_features.index('obj_a_vel' + t_tag[n_instants])] = va
            features[o_id, all_features.index('obj_x_vel' + t_tag[n_instants])] = vx
            features[o_id, all_features.index('obj_y_vel' + t_tag[n_instants])] = vy
            features[o_id, all_features.index('obj_x_size' + t_tag[n_instants])] = o['size_x']
            features[o_id, all_features.index('obj_y_size' + t_tag[n_instants])] = o['size_y']
            features[o_id, all_features.index('obj_dist' + t_tag[n_instants])] = dist
            features[o_id, all_features.index('obj_inv_dist' + t_tag[n_instants])] = 1. - dist  # /(1.+dist*10.)

            max_used_id += 1

        # humans
        for h in data['people']:
            h_id = h['id']

            xpos = h['x'] / 10.
            ypos = h['y'] / 10.
            dist = math.sqrt(xpos ** 2 + ypos ** 2)
            va = h['va'] / 10.
            vx = h['vx'] / 10.
            vy = h['vy'] / 10.
            orientation = h['a']

            if first_frame:
                typeMap[h_id] = 'p'  # 'p' for 'person'
                position_by_id[h_id] = [xpos, ypos, orientation]
                features[h_id, all_features.index('human')] = 1.

                src_nodes.append(h_id)
                dst_nodes.append(room_id)
                edge_types.append(rels.index('p_r'))
                edge_norms.append([1. / len(data['people'])])

                src_nodes.append(room_id)
                dst_nodes.append(h_id)
                edge_types.append(rels.index('r_p'))
                edge_norms.append([1.])

                # Edge features
                edge_features_1[h_id] = th.zeros(n_edge_features)
                edge_features_1[h_id][all_edge_features.index('p_r')] = 1.
                edge_features_2[h_id] = th.zeros(n_edge_features)
                edge_features_2[h_id][all_edge_features.index('r_p')] = 1.

            # Edge features
            t_collision = math.tanh(h['t_collision']/3)
            rx, ry, ra = calculate_relative_position(entity1=(xpos, ypos, orientation), entity2=(0, 0, 0))
            edge_features_1[h_id][all_edge_features.index('x' + t_tag[n_instants])] = rx
            edge_features_1[h_id][all_edge_features.index('y' + t_tag[n_instants])] = ry
            edge_features_1[h_id][all_edge_features.index('orientation' + t_tag[n_instants])] = ra
            edge_features_1[h_id][all_edge_features.index('t_collision' + t_tag[n_instants])] = t_collision

            rx, ry, ra = calculate_relative_position(entity1=(0, 0, 0), entity2=(xpos, ypos, orientation))
            edge_features_2[h_id][all_edge_features.index('x' + t_tag[n_instants])] = rx
            edge_features_2[h_id][all_edge_features.index('y' + t_tag[n_instants])] = ry
            edge_features_2[h_id][all_edge_features.index('orientation' + t_tag[n_instants])] = ra
            edge_features_2[h_id][all_edge_features.index('t_collision' + t_tag[n_instants])] = t_collision

            if last_frame:
                edge_feats_list.append(edge_features_1[h_id])
                edge_feats_list.append(edge_features_2[h_id])

            # Node features
            features[h_id, all_features.index('step_fraction' + t_tag[n_instants])] = step_fraction
            features[h_id, all_features.index('hum_orientation_sin' + t_tag[n_instants])] = math.sin(orientation)
            features[h_id, all_features.index('hum_orientation_cos' + t_tag[n_instants])] = math.cos(orientation)
            features[h_id, all_features.index('hum_x_pos' + t_tag[n_instants])] = xpos
            features[h_id, all_features.index('hum_y_pos' + t_tag[n_instants])] = ypos
            features[h_id, all_features.index('human_a_vel' + t_tag[n_instants])] = va
            features[h_id, all_features.index('human_x_vel' + t_tag[n_instants])] = vx
            features[h_id, all_features.index('human_y_vel' + t_tag[n_instants])] = vy
            features[h_id, all_features.index('hum_dist' + t_tag[n_instants])] = dist
            features[h_id, all_features.index('hum_inv_dist' + t_tag[n_instants])] = 1. - dist  # /(1.+dist*10.)
            features[h_id, all_features.index('t' + str(n_instants))] = 1.

            max_used_id += 1

        # Goal
        goal_id = max_used_id
        max_used_id += 1

        xpos = data['goal'][0]['x'] / 10.
        ypos = data['goal'][0]['y'] / 10.
        dist = math.sqrt(xpos ** 2 + ypos ** 2)

        if first_frame:
            position_by_id[goal_id] = [xpos, ypos, 0.]
            features[goal_id, all_features.index('goal')] = 1
            typeMap[goal_id] = 't'  # 't' for 'goal'
            src_nodes.append(goal_id)
            dst_nodes.append(room_id)
            edge_types.append(rels.index('t_r'))
            edge_norms.append([1.])
            # edge_norms.append([1. / len(data['objects'])])

            src_nodes.append(room_id)
            dst_nodes.append(goal_id)
            edge_types.append(rels.index('r_t'))
            edge_norms.append([1.])

            # Edge features
            edge_features_1[goal_id] = th.zeros(n_edge_features)
            edge_features_1[goal_id][all_edge_features.index('t_r')] = 1.
            edge_features_2[goal_id] = th.zeros(n_edge_features)
            edge_features_2[goal_id][all_edge_features.index('r_t')] = 1.

        # Edge features
        t_collision = math.tanh(data['goal'][0]['t_collision']/3)
        rx, ry, ra = calculate_relative_position(entity1=(xpos, ypos, 0), entity2=(0, 0, 0))
        edge_features_1[goal_id][all_edge_features.index('x' + t_tag[n_instants])] = rx
        edge_features_1[goal_id][all_edge_features.index('y' + t_tag[n_instants])] = ry
        edge_features_1[goal_id][all_edge_features.index('orientation' + t_tag[n_instants])] = ra
        edge_features_1[goal_id][all_edge_features.index('t_collision' + t_tag[n_instants])] = t_collision

        rx, ry, ra = calculate_relative_position(entity1=(0, 0, 0), entity2=(xpos, ypos, 0))
        edge_features_2[goal_id][all_edge_features.index('x' + t_tag[n_instants])] = rx
        edge_features_2[goal_id][all_edge_features.index('y' + t_tag[n_instants])] = ry
        edge_features_2[goal_id][all_edge_features.index('orientation' + t_tag[n_instants])] = ra
        edge_features_2[goal_id][all_edge_features.index('t_collision' + t_tag[n_instants])] = t_collision

        if last_frame:
            edge_feats_list.append(edge_features_1[goal_id])
            edge_feats_list.append(edge_features_2[goal_id])

        # Node features
        features[goal_id, all_features.index('step_fraction' + t_tag[n_instants])] = step_fraction
        features[goal_id, all_features.index('goal_x_pos' + t_tag[n_instants])] = xpos
        features[goal_id, all_features.index('goal_y_pos' + t_tag[n_instants])] = ypos
        features[goal_id, all_features.index('goal_dist' + t_tag[n_instants])] = dist
        features[goal_id, all_features.index('goal_inv_dist' + t_tag[n_instants])] = 1. - dist  # /(1.+dist*10.)
        features[goal_id, all_features.index('t' + str(n_instants))] = 1.

        # Walls
        if not first_frame:
            walls, w_segments, wall_index = generate_walls_information(data, w_segments)

        for iw, wall in enumerate(walls):
            wall_id = max_used_id
            max_used_id += 1

            if first_frame:
                typeMap[wall_id] = 'w'  # 'w' for 'walls'
                position_by_id[wall_id] = [wall.xpos / 1000., wall.ypos / 1000., 0.]
                features[wall_id, all_features.index('wall')] = 1.

                dist = math.sqrt((wall.xpos / 1000.) ** 2 + (wall.ypos / 1000.) ** 2)

                # Links to room node
                src_nodes.append(wall_id)
                dst_nodes.append(room_id)
                edge_types.append(rels.index('w_r'))
                edge_norms.append([1. / len(walls)])

                src_nodes.append(room_id)
                dst_nodes.append(wall_id)
                edge_types.append(rels.index('r_w'))
                edge_norms.append([1.])

                # Edge features
                edge_features_1[wall_id] = th.zeros(n_edge_features)
                edge_features_1[wall_id][all_edge_features.index('w_r')] = 1.
                edge_features_2[wall_id] = th.zeros(n_edge_features)
                edge_features_2[wall_id][all_edge_features.index('r_w')] = 1.

            t_collision = math.tanh(data['walls'][wall_index[iw]]['t_collision'] / 3)
            # Edge features
            rx, ry, ra = calculate_relative_position(entity1=(xpos, ypos, 0), entity2=(0, 0, 0))
            edge_features_1[wall_id][all_edge_features.index('x' + t_tag[n_instants])] = rx
            edge_features_1[wall_id][all_edge_features.index('y' + t_tag[n_instants])] = ry
            edge_features_1[wall_id][all_edge_features.index('orientation' + t_tag[n_instants])] = ra
            edge_features_1[wall_id][all_edge_features.index('t_collision' + t_tag[n_instants])] = t_collision

            rx, ry, ra = calculate_relative_position(entity1=(0, 0, 0), entity2=(xpos, ypos, 0))
            edge_features_2[wall_id][all_edge_features.index('x' + t_tag[n_instants])] = rx
            edge_features_2[wall_id][all_edge_features.index('y' + t_tag[n_instants])] = ry
            edge_features_2[wall_id][all_edge_features.index('orientation' + t_tag[n_instants])] = ra
            edge_features_2[wall_id][all_edge_features.index('t_collision' + t_tag[n_instants])] = t_collision

            if last_frame:
                edge_feats_list.append(edge_features_1[wall_id])
                edge_feats_list.append(edge_features_2[wall_id])

            # Node features
            features[wall_id, all_features.index('step_fraction' + t_tag[n_instants])] = step_fraction
            features[wall_id, all_features.index('wall_orientation_sin' + t_tag[n_instants])] = math.sin(
                wall.orientation)
            features[wall_id, all_features.index('wall_orientation_cos' + t_tag[n_instants])] = math.cos(
                wall.orientation)
            features[wall_id, all_features.index('wall_x_pos' + t_tag[n_instants])] = wall.xpos / 1000.
            features[wall_id, all_features.index('wall_y_pos' + t_tag[n_instants])] = wall.ypos / 1000.
            features[wall_id, all_features.index('wall_dist' + t_tag[n_instants])] = dist
            features[wall_id, all_features.index('wall_inv_dist' + t_tag[n_instants])] = 1. - dist  # 1./(1.+dist*10.)
            features[wall_id, all_features.index('t' + str(n_instants))] = 1.

        n_instants += 1
        first_frame = False

    # interaction links
    for link in data['interaction']:
        typeLdir = typeMap[link['src']] + '_' + typeMap[link['dst']]
        typeLinv = typeMap[link['dst']] + '_' + typeMap[link['src']]

        x_src = position_by_id[link['src']][0]
        y_src = position_by_id[link['src']][1]
        a_src = position_by_id[link['src']][2]

        x_dst = position_by_id[link['dst']][0]
        y_dst = position_by_id[link['dst']][1]
        a_dst = position_by_id[link['src']][2]

        src_nodes.append(link['src'])
        dst_nodes.append(link['dst'])
        edge_types.append(rels.index(typeLdir))
        edge_norms.append([1.])

        src_nodes.append(link['dst'])
        dst_nodes.append(link['src'])
        edge_types.append(rels.index(typeLinv))
        edge_norms.append([1.])

        # Edge features
        rx, ry, ra = calculate_relative_position(entity1=(x_src, y_src, a_src), entity2=(x_dst, y_dst, a_dst))
        edge_features = th.zeros(n_edge_features)
        edge_features[all_edge_features.index(typeLdir)] = 1
        edge_features[all_edge_features.index(link['relation'])] = 1
        edge_features[all_edge_features.index('x' + t_tag[0])] = rx
        edge_features[all_edge_features.index('y' + t_tag[0])] = ry
        edge_features[all_edge_features.index('orientation' + t_tag[0])] = ra
        edge_features[all_edge_features.index('t_collision' + t_tag[0])] = 1.
        edge_feats_list.append(edge_features)

        rx, ry, ra = calculate_relative_position(entity1=(x_dst, y_dst, a_dst), entity2=(x_src, y_src, a_src))
        edge_features = th.zeros(n_edge_features)
        edge_features[all_edge_features.index(typeLinv)] = 1
        edge_features[all_edge_features.index(link['relation'])] = 1
        edge_features[all_edge_features.index('x' + t_tag[0])] = rx
        edge_features[all_edge_features.index('y' + t_tag[0])] = ry
        edge_features[all_edge_features.index('orientation' + t_tag[0])] = ra
        edge_features[all_edge_features.index('t_collision' + t_tag[0])] = 1.
        edge_feats_list.append(edge_features)

    # self edges
    for node_id in range(n_nodes):
        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index('self'))
        edge_norms.append([1.])

        # Edge features
        edge_features = th.zeros(n_edge_features)
        edge_features[all_edge_features.index('self')] = 1
        edge_feats_list.append(edge_features)

    # Convert outputs to tensors
    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    edge_feats = th.stack(edge_feats_list)

    return src_nodes, dst_nodes, n_nodes, features, edge_feats, edge_types, edge_norms, position_by_id, typeMap, \
           []
