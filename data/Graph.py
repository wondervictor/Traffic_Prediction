# -*- coding: utf-8 -*-

class Graph(object):

    def __init__(self, value, linked=[None]):
        self.value = value
        self.linked = linked

    def get_linked(self):
        return self.linked

    def has_subnodes(self):
        if len(self.linked) > 0:
            return True
        else:
            return False


class Vertex(object):
    def __init__(self, value, vid):
        self.value = value
        self.id = vid


def create_edges(graph_list):
    edges = []
    vid = 0
    for graph in graph_list:
        ver = Vertex(graph.value, vid)
        vid += 1
        edges.append(ver)


def read_from_csv():
    nodes = dict()
    with open('graph.csv', 'r') as f:
        for line in f.readlines():
            pairs = line.replace('\n','').split(',')
            sub_nodes = []
            key = pairs[0]
            if key in nodes:
                sub_nodes = nodes[key]
            sub_nodes.append(pairs[1])
            if pairs[1] not in nodes:
                nodes[pairs[1]] = []
            nodes[key] = sub_nodes
    graph_list = []

    for key in nodes:
        node = Graph(key, nodes[key])
        graph_list.append(node)

""" Graph Algorithms"""


def bfs():
    pass

