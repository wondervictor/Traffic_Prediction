# -*- coding:utf-8 -*-

# 有向图转邻接矩阵


def convert(node_dict):

    r = len(node_dict)
    nodes = node_dict.keys()
    with open('graph_mat.csv', 'a+') as f:
        title = " ,"
        title += ",".join(nodes)
        title += '\n'
        f.write(title)
        for key in node_dict:
            line_node = ['0']*r
            sub_nodes = node_dict[key]
            for sub_node in sub_nodes:
                idx = nodes.index(sub_node)
                line_node[idx] = '1'
            line = key + ","
            line += ",".join(line_node)
            line += '\n'
            f.write(line)


def get_data(filename):
    link_pairs = []
    with open(filename, 'r') as f:
        for pair in f.readlines():
            pairs = pair.replace("\n", "").split(",")
            link_pair = (pairs[0], pairs[1])
            link_pairs.append(link_pair)

    node = dict()
    for pair in link_pairs:
        nodes = []
        if pair[0] in node:
            nodes = node[pair[0]]
        new_nodes = nodes
        new_nodes.append(pair[1])
        node[pair[0]] = new_nodes
        new_nodes = []
        if pair[1] not in node:
            node[pair[1]] = new_nodes
    return node

#node_dict = get_data('graph.csv')
#convert(node_dict)


def points_to_point():
    all_points = {}

    with open('graph.csv', 'r') as f:
        for line in f.readlines():
            points = line.replace('\n', '').split(',')
            point_a = points[0]
            point_b = points[1]

            if point_a in all_points:
                __points = all_points[point_a]
                if point_b not in __points:
                    __points.append(point_b)
                all_points[point_a] = __points
            else:
                all_points[point_a] = [point_b]
            if point_b in all_points:
                __points = all_points[point_b]
                if point_a not in __points:
                    __points.append(point_a)
                all_points[point_b] = __points
            else:
                all_points[point_b] = [point_a]

    with open('Points.txt', 'a+') as f:
        for key in all_points:
            line = "%s;" % key
            line += ','.join(all_points[key])
            line += '\n'
            f.write(line)


def select_point_to_test(points, point_name):
    data = {}
    with open('speeds_without_zero.csv', 'r') as f:
        for line in f.readlines()[1:]:
            line_elements = map(int, line.replace('\n', '').split(','))
            if line_elements[0] in points or line_elements[0] == point_name:
                print '---> %s' % line_elements[0]
                data[line_elements[0]] = line_elements[1:]

    with open('speed_data/%s.txt' % point_name, 'a+') as f:
        line = ",".join(['%s' % i for i in data[point_name]])
        line += '\n'
        f.write(line)
        print '<----%s---->' % point_name
        for key in points:
            line = ",".join(['%s' % i for i in data[key]])
            line += '\n'
            f.write(line)
            print key
