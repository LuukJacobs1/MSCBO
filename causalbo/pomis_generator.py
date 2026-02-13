from networkx import DiGraph, ancestors, topological_sort, strongly_connected_components, descendants, shortest_path_length
from itertools import combinations
from collections import deque
import networkx as nx

#SCM and do-calculus operations
class PomisGenerator():
    def __init__(self, graph: DiGraph | list, exploration_set: list[str], output_node: str):
        if type(graph) is DiGraph:
            self.graph = graph
        else:
            try:
                self.graph = DiGraph(graph)
            except:
                raise Exception('Graph must be networkx.DiGraph object or networkx.DiGraph formatted list.')
        self.output_node = output_node
        self.interventional_domain = exploration_set
        self.pomis = None

    # Collect parent nodes for 1 or more node(s)
    def pa(self, G: DiGraph, muct: list[str]):
        return {parent for x in muct for parent in G.predecessors(x) if parent in self.interventional_domain.keys()}

    # Collect the c-component (strongly connected) containing node X within graph G
    def CC(self, G: DiGraph, X: str):

        H = G.copy()
        
        # Replace unobserved confounders with bidirectional edges
        for unobserved_node in set(H.nodes) - self.interventional_domain.keys() - set(self.output_node):
            children = list(H.successors(unobserved_node))
            H.remove_edges_from([(unobserved_node, child) for child in children])
            H.remove_node(unobserved_node)
            # Add bidirectional edges between each pair of children
            for (child1, child2) in combinations(children, 2):
                H.add_edge(child1, child2)
                H.add_edge(child2, child1)

        # Find all strongly connected components in the bidirected subgraph
        sccs = strongly_connected_components(H)

        # Find and return the component that contains X
        for component in sccs:
            if X in component:
                return component
        
    # Calculate the minimal Unobserved Confounders Territory
    def MUCT(self, G: DiGraph):
        H = G.subgraph(ancestors(G, self.output_node).union({self.output_node}))

        Qs = {self.output_node}
        Ts = frozenset({self.output_node})

        while Qs:
            Q1 = Qs.pop()
            Ws = self.CC(H, Q1)     
            if Ws:
                Ts |= Ws 
                Qs = set.union(Qs, *(descendants(H, w) for w in Ws)) - Ts         
            else:
                Qs -= Ts
        return Ts

    # Calculate MUCT (see MUCT) and IB (interventional border), which represents the parents of the MUCT
    def MUCT_IB(self, G: DiGraph):
        Zs = self.MUCT(G)
        return Zs, self.pa(G, Zs) - Zs

    # Recursively calculates the POMIS for a given Graph G
    def subPOMIS(self, G: DiGraph, Ws: list[str], obs = None):
        if obs is None:
            obs = set()
        out = []

        for i, W_i in enumerate(Ws):
            H = G.copy()
            # Perform intervention / do
            edges_to_remove = set(H.in_edges(W_i))
            H.remove_edges_from(edges_to_remove)

            # Calculate minimal Unobserved Confounders Territory and Interventional Border
            Ts, Xs = self.MUCT_IB(H)
            new_obs = obs | set(Ws[:i])
            
            if not (Xs & new_obs):
                out.append(Xs)
                new_Ws = [w for w in Ws[i + 1:] if w in (Ts & self.interventional_domain.keys())]
                
                if new_Ws:
                    edges_to_remove = set()
                    for x in Xs:
                        edges_to_remove += set(H.in_edges(x))

                    H = H.remove_edges_from(edges_to_remove)
                    out.extend(self.subPOMIS(H, new_Ws, new_obs))

        return {frozenset(_) for _ in out}

    # Collect all POMIS (Potentialy optimal minimal intervention sets) for the input graph with respect to its output node
    def POMIS(self):
        # Only retain Ancestors of the response node
        H = self.graph.subgraph(ancestors(self.graph, self.output_node).union({self.output_node})).copy()

        # Calculate minimal Unobserved Confounders Territory and Interventional Border
        muct, ib = self.MUCT_IB(H)

        # Perform intervention of the interventional border
        edges_to_remove = frozenset()
        for x in ib:
            edges_to_remove |= set(H.in_edges(x))
        H.remove_edges_from(edges_to_remove)

        reversed_nodes = deque()
        # Perform a reverse topological sort and "climb" the network in a Depth-First manner
        for node in topological_sort(H):
            if node != self.output_node:
                reversed_nodes.appendleft(node)

        return list(self.subPOMIS(H, [n for n in reversed_nodes if n in list((muct | (ib)) - {self.output_node}) & self.interventional_domain.keys()]) | {frozenset(ib)})   
    
    # Calculate the total distance to the query variable for each POMIS and return
    def calculate_distances(self, pomis):
        return [sum([shortest_path_length(G = self.graph, source = n, target = self.output_node) for n in pom])  for pom in pomis]
