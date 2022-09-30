from utils.graph import Graph, DirectedEdge


class ObjectTypeGraph(Graph):

    def __init__(self, otypes):
        self.otypes = otypes
        Graph.__init__(self)

    def add_nodes_by_names(self, node_names):
        Graph.add_nodes_by_name(self, node_names)

    def add_edge_by_names(self, source, target):
        source_node = list(filter(lambda node: node.name == source, self.nodes))[0]
        target_node = list(filter(lambda node: node.name == target, self.nodes))[0]
        self.edges += [DirectedEdge(source_node, target_node)]

    def make_info(self):
        self.__make_shortest_paths()
        self.__make_neighbor_otypes()
        self.__make_component_splits()

    def has_edge(self, source_name, target_name):
        return any(edge.source.name == source_name and edge.target.name == target_name for edge in self.edges)

    def get_parents(self, node_name):
        return list(map(lambda edge: edge.source,
                        list(filter(lambda edge: edge.target.name == node_name, self.edges))))

    def get_children(self, node_name):
        return list(map(lambda edge: edge.target,
                        list(filter(lambda edge: edge.source.name == node_name, self.edges))))

    def __make_neighbor_otypes(self):
        self.neighborOtypes = dict()
        for otype in self.otypes:
            self.neighborOtypes[otype] = {
                "parents": list(map(lambda node: node.name, self.get_parents(otype))),
                "children": list(map(lambda node: node.name, self.get_children(otype)))
            }

    def get_neighbor_otypes(self, otype):
        return self.neighborOtypes[otype]

    def __make_shortest_paths(self):
        shortest_paths = {}
        for otype in self.otypes:
            level = 0
            current_path = []
            buffer = [(otype, current_path, level)]
            otype_shortest_paths = dict()
            while len(buffer) > 0:
                current_otype, current_path, current_level = buffer[0]
                buffer = buffer[1:]
                if current_otype not in otype_shortest_paths:
                    otype_shortest_paths[current_otype] = current_path
                    otype_parents = self.get_parents(current_otype)
                    otype_children = self.get_children(current_otype)
                    neighbor_names = list(map(lambda node: node.name,
                                              otype_parents + otype_children))
                    buffer += [(next_otype, current_path + [current_otype], current_level + 1) for next_otype in
                               neighbor_names
                               if not next_otype in otype_shortest_paths]
            shortest_paths[otype] = otype_shortest_paths
        self.shortest_paths = shortest_paths

    def get_component_split(self, ot1, ot2):
        return self.componentSplits[(ot1, ot2)]

    def __make_component_splits(self):
        self.componentSplits = {}
        for ot1 in self.otypes:
            neighbors = self.neighborOtypes[ot1]
            for ot2 in neighbors["children"]:
                ot1_side = []
                ot2_side = []
                for ot in [ot for ot in self.otypes if ot != ot1 and ot != ot2]:
                    if ot not in self.shortest_paths[ot1]:
                        continue
                    sp1 = self.shortest_paths[ot1][ot]
                    sp2 = self.shortest_paths[ot2][ot]
                    if len(sp1) < len(sp2):
                        ot1_side.append(ot)
                    elif len(sp1) > len(sp2):
                        ot2_side.append(ot)
                    else:
                        raise ValueError(
                            "The paths from " + ot1 + " and from " + ot2 + " to " + ot + " in the object type graph" + \
                            " have the same length.")
                self.componentSplits[(ot1, ot2)] = (ot1_side, ot2_side)
                self.componentSplits[(ot2, ot1)] = (ot2_side, ot1_side)
