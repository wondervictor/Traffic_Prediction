# -*- coding: utf-8 -*-

"""
extend the surrounded node ranges.

1 -> 2 ->center ->3 -> 4
"""

def create_link_list(filename):
    graph = {}
    all_nodes = {}
    with open(filename, 'r') as open_file:
        for line in open_file.readlines():
            pair_points = map(int, line.rstrip('\r\n').split(','))
            key = '%s' % (pair_points[0])
            if key in graph:
                point_list = graph[key]
                point_list.append(pair_points[1])
                graph[key] = point_list
            else:
                graph[key] = [pair_points[1]]
            key2 = '%s' % pair_points[1]
            if key2 in graph:
                point_list = graph[key2]
                point_list.append(pair_points[0])
                graph[key2] = point_list
            else:
                graph[key2] = [pair_points[0]]

    for key in graph:
        point_list = graph[key]
        nodes = {}
        center_node = int(key)
        nodes['0'] = center_node
        nodes['1'] = point_list
        subnodes = []
        for point in point_list:
            key = '%s' % point
            if key in graph:
                subpoint_list = graph['%s' % point ]
                subnodes.extend(subpoint_list)
        while center_node in subnodes:
            subnodes.remove(center_node)
        nodes['2'] = subnodes
        all_nodes[key] = nodes

    with open('two_dist_point', 'w+') as writingFile:
        for key in all_nodes:
            value = graph[key]
            line = '%s;' % key
            line += ','.join(['%s' % k for k in value['1']])
            line += ';'
            line += ','.join(['%s' % k for k in value['2']])
            line += '\n'
            writingFile.write(line)

if __name__ == '__main__':
    create_link_list('graph.csv')