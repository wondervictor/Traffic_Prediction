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

node_dict = get_data('graph.csv')

convert(node_dict)




