import os
import pickle

from utils.graph import Graph, DirectedEdge


class ObjectTypeGraph(Graph):

    @classmethod
    def load(cls, session_path):
        otg_path = os.path.join(session_path, "object_type_graph.pkl")
        return pickle.load(open(otg_path, "rb"))

    def __init__(self, otypes):
        self.otypes = otypes
        Graph.__init__(self)

    def add_nodes_by_names(self, node_names):
        Graph.add_nodes_by_name(self, node_names)

    def add_edge_by_names(self, source, target):
        source_node = list(filter(lambda node: node.name == source, self.nodes))[0]
        target_node = list(filter(lambda node: node.name == target, self.nodes))[0]
        self.edges += [DirectedEdge(source_node, target_node)]

    def has_edge(self, source_name, target_name):
        return any(edge.source.name == source_name and edge.target.name == target_name for edge in self.edges)

    def get_neighbors(self, node_name):
        return list(set(
            list(map(lambda edge: edge.target.name,
                    list(filter(lambda edge: edge.source.name == node_name, self.edges)))) +\
            list(map(lambda edge: edge.source.name,
                    list(filter(lambda edge: edge.target.name == node_name, self.edges))))
        ))
