import numpy as np
import networkx as nx
import networkx.algorithms.tree.mst as mst
import matplotlib.pyplot as plt


from itertools import product

# def genbipartite_foralternate(Graph):
#     Right_hist = "right" # Graph.number_of_nodes()
#     Bipartite = nx.Graph()    

#     for v in Graph.nodes:
#         Bipartite.add_nodes_from( [v , (v,Right_hist) ] )
#         Bipartite.add_edge(v , (v,Right_hist), weight=0 )
#         Bipartite.edges[v , (v,Right_hist)]["fixed"] = mst.EdgePartition.INCLUDED 
#         Bipartite.edges[v , (v,Right_hist)]["weight"] = 0 
    
#     for (u, v, weight) in Graph.edges.data('weight'):
#         Bipartite.add_edge(v , (u,Right_hist)) 
#         Bipartite.edges[v , (u,Right_hist)]["weight"] = weight 
#         Bipartite.edges[v , (u,Right_hist)]["real"] = True
#         if 'fixed' in  Graph.edges[v,u]:
#             Bipartite.edges[v , (u,Right_hist)]['fixed'] = Graph.edges[v,u]['fixed']
#         else:
#             Bipartite.edges[v , (u,Right_hist)]['fixed'] = mst.EdgePartition.OPEN

#     return Bipartite

def product_graph(Graph, group, weight_function):
    
    # indecators = []

    Lift = nx.Graph()
    for v in Graph.nodes(): 
        for g in group:
            Lift.add_node(("left", (g,v)))

    def add_unreal_edge(edge):
        Lift.add_edge(*edge)
        Lift.edges[edge]['weight']  = 0
        Lift.edges[edge]['fixed']   = mst.EdgePartition.INCLUDED
        Lift.edges[edge]['real']    = False

    for v in Graph.nodes(): 
        for g in group[1:]:
            edge = (("left", (group[0],v)), ("left", (g,v)))   
            add_unreal_edge(edge)    
    
    for u in Graph.nodes():
        for v in Graph.nodes(): 
            for g in group:
                for h in group:
                    Lift.add_node(("right", (g,v,h,u)))
                    edge = (("left", (h,u)), ("right", (g,v,h,u)))   
                    add_unreal_edge(edge) 
    
    for u in Graph.nodes(): 
        for v in Graph.nodes(): 
            for g in group[1:]:
                for h in group:
                    edge = (("right", (group[0],v,h,u)), ("right", (g,v,h,u)))   
                    add_unreal_edge(edge)
             
    for g, h in product(group,group):
        for (u, v, weight) in Graph.edges.data('weight'):
            edge = (("left", (g,v)), ("right", (g,v,h,u)))
            Lift.add_edge(*edge)
            Lift.edges[edge]['weight']  = weight_function(g,v,h,u)
            Lift.edges[edge]['fixed']   = mst.EdgePartition.OPEN
            Lift.edges[edge]['real']    = True
    
    return Lift

def MST( Bipartite ):
    return mst.kruskal_mst_edges(Bipartite, True, partition="fixed" )

def extract_circuit(G):
    edges = MST(G)
    ret = []
    for e in edges:
        if 'real' in e[-1] and e[-1]['real']: 
            g,v,h,u = e[1][1]
            ret.append((g,v,h,u))
    return ret
if __name__ == "__main__":
    # G = nx.petersen_graph()
    G = nx.Graph()
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_node(4)
    G.add_edge(1,2)
    G.add_edge(1,3)
    G.add_edge(1,4)
    G.add_edge(2,3)
    G.add_edge(2,4)
    # subax1 = plt.subplot(121)
    for u,v in G.edges:
        G.edges[u,v]['weight'] = 1
    def w_func(x,y):
        return { 
            (1,1): 1,
            (1,2): 0,
            (2,1): 1,
            (2,2): 1
        }[(x,y)]
    
    def w_func_func(g,v,h,u):
        return w_func(g,h)

    G = product_graph(G, [1,2],  w_func_func)
    # G = genbipartite_foralternate(G)

    print(extract_circuit(G))
    exit(0) 

    subax1 = plt.subplot(121)
    options = {
        'node_color': 'black',
        'node_size': 10,
        'width': 0.5,
    }

    pos = nx.spring_layout(G, seed=50)
    nx.draw_networkx_nodes(G, pos, node_size=10)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, edge_color="b")# style="dashed"


    edge_labels = nx.get_edge_attributes(G, "fixed")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=5)

    edgs = MST(G)
    print(len(list(G.edges)))
    # print(len(list(edgs)))
    E, E2 = [], []
    for e in edgs:
        if 'real' in e[-1] and e[-1]['real']: 
            # if e[-1] 
            E.append(e)
        # else:
        #     E2.append(e)
        # print(e)
    # E = []
    subax2 = plt.subplot(122)
    nx.draw_networkx_edges(G, pos, edgelist = E, width=0.5, alpha=0.5, edge_color="b")# style="dashed"
    nx.draw_networkx_edges(G, pos, edgelist = E2, width=0.5, alpha=0.5, edge_color="r")# style="dashed"

    # for u,v in edgs:
    #     if edgs[u,v]["fixed"] !=

    # plt.show()