import networkx as nx 
from itertools import combinations
from matplotlib import pyplot as plt

from random import choice

if __name__ == "__main__":
    

    S = [ "".join( [choice(["X", "Z", "Y", "I"]) for _ in range(10)] ) for __ in range(276)]

    for s in sorted(S):
        print(s)
        

    # n = 4
    # G = nx.Graph()

    # # pairs = combinations(list(range(10))

    
    # for (i,j) in combinations(list(range(10)), r = 2):
    #     G.add_node((i,j))
    

    # prev = list(G.nodes())[0]
    # first = prev
    # for node in  list(G.nodes())[1:]:
    #     G.add_edge(prev, node)
    #     prev = node
    # G.add_edge(prev, first)
    

    # subax1 = plt.subplot(121)
    # options = {
    #     'node_color': 'black',
    #     'node_size': 10,
    #     'width': 0.5,
    # }

    # pos = nx.spring_layout(G, seed=50)
    # nx.draw_networkx_nodes(G, pos, node_size=10)
    # nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, edge_color="b")# style="dashed"
    # nx.draw_networkx_labels(G,pos, font_size=6)
    # plt.show()