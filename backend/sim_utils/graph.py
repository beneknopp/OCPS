class Graph:

    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node):
        self.nodes.append(Node)

    def add_nodes_by_name(self, node_names):
        self.nodes += [Node(name) for name in node_names]

    def add_edge(self, edge):
        self.nodes = list(set(self.nodes + [edge.source] + [edge.target]))
        self.edges.append(edge)


class Node:

    def __init__(self, name):
        self.name = name
        self.outgoing_edges = []
        self.incoming_edges = []


class DirectedEdge:

    def __init__(self, source, target):
        self.source = source
        self.target = target
        source.outgoing_edges.append(self)
        target.incoming_edges.append(self)
