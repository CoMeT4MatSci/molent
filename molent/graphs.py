import numpy as np
import networkx as nx

from rdkit import Chem
from scipy.optimize import linear_sum_assignment

def mol2graph(m, root=None):
    """ Constructs a Networkx graph from an rdkit molecule.
        based on code by Jan H. Jensen (https://github.com/jensengroup/GED)
    
        Args:
            m(rdchem.m): rdkit mecule.
            root(int or None): Indiates which atom should be used as root node of the graph
        Returns:
            nx.Graph: Graph of the mecule with bond orders as edge weights and the atomic 
                      numbers as node attribute 'atom'
    """

    # construct graph from adjacency matrix and put bond orders as edge weights
    am = Chem.GetAdjacencyMatrix(m, useBO=True)    
    G = nx.from_numpy_array(am)
    
    # set node attribute 'atom' to atomic number
    for i,atom in enumerate(m.GetAtoms()):
        G.nodes[i]['atom'] = atom.GetAtomicNum()

    # optionally, relabel the nodes such that root is the first node
    if root is not None:
        node_map = {0: root, root: 0}
        G = nx.relabel_nodes(G, node_map, copy=True)
        
    return G


def graphedit_similarity(graphs, root_equality=False, timeout=None):
    """ Calculates the similarity matrix from the pairwise graph edit distances of
        a list of graphs.
    """
    S = np.zeros((len(graphs), len(graphs)))    
    for i in range(len(graphs)):
        G1 = graphs[i]
        G1_size = G1.number_of_edges() + G1.number_of_nodes()
        
        for j in range(i,len(graphs)):
            G2 = graphs[j]
            max_d = max(G1_size, G2.number_of_edges() + G2.number_of_nodes())
            GED = nx.graph_edit_distance(G1, G2, 
                                         node_match=lambda a,b: a['atom'] == b['atom'], 
                                         edge_match=lambda a,b: a['weight'] == b['weight'], 
                                         roots=(0, 0), timeout=timeout)
            
            if max_d > GED:
                S[i,j] = np.exp(-GED/(max_d-GED))
            else:
                S[i,j] = 0.0
            
            if root_equality:
                S[i,j] = S[i,j] * int(G1.nodes[0]['atom']==G2.nodes[0]['atom'])
            
            S[j,i] = S[i,j]
    return S


def bp_graphedit_similarity(graphs, root_equality=False, timeout=None):
    """ Calculates the similarity matrix from the pairwise approximated graph edit distances of
        a list of graphs.
    """
    S = np.zeros((len(graphs), len(graphs)))    
    for i in range(len(graphs)):
        G1 = graphs[i]
        G1_size = G1.number_of_edges() + G1.number_of_nodes()
        
        for j in range(i,len(graphs)):
            G2 = graphs[j]
            max_d = max(G1_size, G2.number_of_edges() + G2.number_of_nodes())
            GED = ged(G1, G2, node_sub=lambda a,b: 1-int(a['atom'] == b['atom']), 
                              edge_sub=lambda a,b: (1-int(a['weight'] == b['weight']))/2,
                              roots=(0, 0))
            
            if max_d > GED:
                S[i,j] = np.exp(-GED/(max_d-GED))
            else:
                S[i,j] = 0.0
            
            if root_equality:
                S[i,j] = S[i,j] * int(G1.nodes[0]['atom']==G2.nodes[0]['atom'])
            
            S[j,i] = S[i,j]
    return S


"""
    Implementation of an approximate graph edit distance.
    (partly inspired by https://github.com/priba/aproximated_ged)

    Kaspar Riesen, and Horst Bunke. "Approximate graph edit distance computation by means of bipartite graph matching."
    Image and Vision computing 27.7 (2009): 950-959.
    https://doi.org/10.1016/j.imavis.2008.04.004

    Riesen, Kaspar. "Bipartite Graph Edit Distance" 
    In: "Structural Pattern Recognition with Graph Edit Distance." 
    Advances in Computer Vision and Pattern Recognition. Springer, Cham. (2015)
    https://doi.org/10.1007/978-3-319-27252-8_3
"""
def nodes_insertion_costs(G, node_ins=1):
    return [node_ins]*G.number_of_nodes()

def nodes_deletion_costs(G, node_del=1):
    return [node_del]*G.number_of_nodes()

def nodes_substitution_costs(G1, G2, node_sub=None):
    CM = np.zeros([G1.number_of_nodes(),G2.number_of_nodes()])
    if node_sub:    
        for i1,N1 in enumerate(G1.nodes()):
            for i2,N2 in enumerate(G2.nodes()):
                    CM[i1,i2] = node_sub(G1.nodes[N1], G2.nodes[N2])
                
    return CM

def edges_insertion_costs(G, edge_ins=1):
    # number of neighbors = number of edges for each node
    node_neigh = np.array([v for i,v in nx.degree(G)])
    
    return node_neigh*edge_ins

def edges_deletion_costs(G, edge_del=1):
    # number of neighbors = number of edges for each node
    node_neigh = np.array([v for i,v in nx.degree(G)])
    
    return node_neigh*edge_del

def edge_substitution_costs(N1adj, N2adj, edge_sub=None):
    CM = np.zeros([len(N1adj),len(N2adj)])
    
    if edge_sub:
        for i1,N1 in enumerate(N1adj.items()):
            for i2,N2 in enumerate(N2adj.items()):
                    CM[i1,i2] = edge_sub(N1[1], N2[1])
                
    return CM

def edge_ed_costs(N1adj, N2adj, edge_ins=1, edge_del=1, edge_sub=None):
    # Compute edge edit cost matrix
    CM = edge_cost_matrix(N1adj, N2adj, edge_ins=edge_ins, edge_del=edge_del, 
                                   edge_sub=edge_sub)

    # Munkres algorithm
    row_ind, col_ind = linear_sum_assignment(CM)

    # Edit distance for edges
    dist = CM[row_ind, col_ind].sum()

    return dist
    
def edge_cost_matrix(N1adj, N2adj, edge_ins=1, edge_del=1, edge_sub=None):
    n1 = len(N1adj)
    n2 = len(N2adj)
    
    # Cost matrix
    CM = np.zeros([n1+n2,n1+n2])

    # Insertion costs (lower left)
    CM[n1:, 0:n2] = np.inf
    np.fill_diagonal(CM[n1:, 0:n2], [edge_ins]*len(N2adj))

    # Deletion costs (upper right)
    CM[0:n1, n2:] = np.inf
    np.fill_diagonal(CM[0:n1, n2:], [edge_del]*len(N1adj))

    # Substitution costs
    CM[0:n1, 0:n2] = edge_substitution_costs(N1adj, N2adj, edge_sub=edge_sub)
    
    return CM

def cost_matrix(G1, G2, node_ins=1, node_del=1, node_sub=None, edge_ins=1, edge_del=1, edge_sub=None, roots=None):
    
    n1 = G1.number_of_nodes()
    n2 = G2.number_of_nodes()
    
    # Cost matrix
    CM = np.zeros([n1+n2,n1+n2])

    # Insertion costs (lower left)
    CM[n1:, 0:n2] = np.inf
    np.fill_diagonal(CM[n1:, 0:n2], nodes_insertion_costs(G2, node_ins=node_ins) + edges_insertion_costs(G2, edge_ins=edge_ins))

    # Deletion costs (upper right)
    CM[0:n1, n2:] = np.inf
    np.fill_diagonal(CM[0:n1, n2:], nodes_deletion_costs(G1, node_del=node_del) + edges_deletion_costs(G1, edge_del=edge_del))

    # Substitution costs
    # ... first, from nodes
    node_subst = nodes_substitution_costs(G1, G2, node_sub=node_sub)

    # ... then, from edges
    for i1, N1 in enumerate(G1.nodes()):
        for i2, N2 in enumerate(G2.nodes()):
            node_subst[i1, i2] += edge_ed_costs(G1[N1], G2[N2], edge_ins=edge_ins, edge_del=edge_del, 
                                               edge_sub=edge_sub)

    # upper left
    CM[0:n1, 0:n2] = node_subst

    if roots:
        root_1, root_2 = roots
        root_1_idx, root_2_idx = list(G1.nodes).index(root_1), list(G2.nodes).index(root_2)

        # substitution root_1->root_2 is enforced
        cm12 = CM[root_1_idx, root_2_idx]
        CM[root_1_idx,:] = np.inf
        CM[:,root_2_idx] = np.inf
        CM[root_1_idx, root_2_idx] = cm12
    
    return CM

def ged(G1, G2, node_ins=1, node_del=1, node_sub=None, edge_ins=1, edge_del=1, edge_sub=None, roots=None):
    """
        Calculate approximate graph edit distance between two graphs.

        Kaspar Riesen, and Horst Bunke. "Approximate graph edit distance computation by means of bipartite graph matching."
        Image and Vision computing 27.7 (2009): 950-959.
        https://doi.org/10.1016/j.imavis.2008.04.004

        Riesen, Kaspar. "Bipartite Graph Edit Distance" 
        In: "Structural Pattern Recognition with Graph Edit Distance." 
        Advances in Computer Vision and Pattern Recognition. Springer, Cham. (2015)
        https://doi.org/10.1007/978-3-319-27252-8_3
    """
    # Compute cost matrix
    CM = cost_matrix(G1, G2, node_ins=node_ins, node_del=node_del, node_sub=node_sub, 
                     edge_ins=edge_ins, edge_del=edge_del, edge_sub=edge_sub, roots=roots)

    # Munkres algorithm
    row_ind, col_ind = linear_sum_assignment(CM)

    # Graph edit distance
    dist = CM[row_ind, col_ind].sum()

    return dist