from copy import deepcopy
from turtle import color
import networkx as nx
from numpy import argmin, average
from sympy import permutedims
from Hamiltonian_parser import parser, local_Hamiltonian,genreate_circut,genreate_optimzed_circut,SWAPlocal_Hamiltonian
from itertools import permutations, product, combinations

from random import sample, shuffle
from  tqdm import tqdm
import pickle as pkl
from datetime import datetime
import numpy as np

from matplotlib import pyplot as plt


class Permutation_Base():
    def __init__(self, arr) -> None:
        self.arr = arr
        self.parent = self


def generated_the_product_graph(num_of_terms=20):
    terms = parser() #sample(parser(), num_of_terms) #parser()# sample(parser(), num_of_terms)
    def generated_the_product_graph_by_base(_terms, number_premu=0):
        
        G        = nx.Graph()
        Gproduct = nx.Graph()

        def check_solid_edge(H1, H2, H3, H4):
            res = [Hi.solid_product(Hj)[0] for Hi,Hj in [ (H1,H2), (H1,H4), (H2, H3), (H3, H4)]]
            return all([x for x in res])

        def check_edge(H1, H2, H3, H4):
            if H1.dis(H2) <= 1 and H1.dis(H4) <= 1 and  H2.dis(H3) <= 1 and H3.dis(H4) <= 1:
                return True, max(H1.dis(H3), H2.dis(H4))
            else :
                return False, None 
        
        edges_set = set()
        print(len(_terms))
        for (H1, H2) in product(_terms, _terms):            
            for H in _terms:
                if H1.solid_product(H2)[0]:
                    G.add_edge(H1, H2)
                    edges_set.add( (H1,H2) )
                    edges_set.add( (H2,H1) )
            # if H1.tensorspace(H2):
            #     vertices.append((H1, H2))
        print("hi")
        for e in G.edges():
            H1,H2 = e
            for H3,H4 in \
                product(list(G.adj[H1]),list(G.adj[H2])):  
                if (H1,H4) in edges_set and\
                    (H3,H4) in edges_set and\
                        (H2,H3) in edges_set:
                    Gproduct.add_edge((H1,H2), (H3, H4))

                    # exist, cost = check_edge(H1, H2, H3, H4)
                    Gproduct.edges[(H1, H2), (H3, H4)]['weight'] = 1
                # if check_solid_edge(H1, H2, H3, H4):
                    Gproduct.edges[(H1, H2), (H3, H4)]['solid'] = True 
                    Gproduct.edges[(H1, H2), (H3, H4)]["permutation"] = j
                
        print("hi")
        print(f"vertices:{Gproduct.number_of_nodes()}\t edges: ~{Gproduct.number_of_edges()}")
        return Gproduct, _terms     
    

    # return pkl.load( open(f"mainG.pkl-276-1", "br"))
            

    permutations = list(map(lambda x: Permutation_Base(x) , [
        # [0,2,4,6,8,1,3,5,7,9],
        [0,1,2,3,4,5,6,7,8,9]
        # [0,7,4,6,8,9,3,5,2,1]
    ]))
    graphs = []
    perm_terms = []
    mainG = nx.Graph()
    for j, permutation in enumerate(permutations): 
        perm_terms = list(map( lambda x : x.newbase(permutation.arr), terms))
        # print(perm_terms)
        G, _ = generated_the_product_graph_by_base(perm_terms, number_premu=j) 
        
        # graphs.append(G)
        mainG = nx.compose(mainG, G)

    pkl.dump((mainG, terms ,permutations), open(f"mainG.pkl-{len(terms)}-{len(permutations)}", "bw+"))
    return mainG, terms, permutations
from random import shuffle, choice

def select(_list, v, G, flag = True):
    solids = list(filter(lambda u : G.edges[v,u]['solid'], _list))
    if len(solids) > 0:
        return choice(solids)
    if flag:
        return choice(_list)
    else:
        return []
#randomized DFS.
def sample_path(G, terms) -> tuple((nx.Graph, set)):
    print("sample_path")
    color = set()
    
    def DFS(v, T, _color, sign=0, flag =True):        
        term1, term2 = v
        
        if (term1.parent in _color) or (term2.parent in _color):
            return  _color, sign
        
        _color.add(term1.parent)
        _color.add(term2.parent)
        can_packed = [None]
        while len(can_packed) > 0 :
            
            can_packed = list(filter(lambda x :\
                (x[0].parent not in _color) and\
                    (x[1].parent not in _color), G.adj[v] ))
            
            if len(can_packed) > 0:
                u =  select( can_packed, v, G, flag=flag)
                T.add_edge(v,u)
                T.edges[v,u]['permutation'] = G.edges[v,u]['permutation']
                T.edges[v,u]['sign'] = sign
                _color, sign = DFS(u, T, _color, sign=sign+1, flag=flag)
        return _color, sign
    T = nx.DiGraph()
    l = []

    for v in G.nodes():
        T.add_node(v)
        l.append(v)
    
    
    last_size = T.number_of_edges()
    _sign = 0
    for v in G.nodes(): 
        color, _sign = DFS(v, T, color, sign=_sign, flag=False)
        # if last_size == T.number_of_edges():
        #     for u in [v[0], v[1]]:
        #         if u.parent in color:
        #             color.remove(u.parent)

        last_size = len(T.edges.values())
    # for v in G.nodes(): 
    #     color = DFS(v, T, color, flag=True)

    return T, color




def get_Diameter(Tree: nx.Graph) -> tuple((tuple(([], int)),tuple(([], int)))) :

    def DFS_tree_depth(G, v):
        
        if len(list(G.adj[v])) == 0 :
            return (([v],1),([v],1))
        
        branches = []
        maxinnerpath, maxinnerdepth = [],0
        for u in list(G.adj[v]):
            ((temppath, tempdepth), \
                (tempinnerpath, tempinnerdepth)) = DFS_tree_depth(G,u)
            
            if maxinnerdepth <  tempinnerdepth:
                maxinnerpath, maxinnerdepth = tempinnerpath, tempinnerdepth
            
            branches.append((temppath, tempdepth))
        
        maxpath, maxdepth = [],0 
        
        for (b1, d1),(b2, d2) in combinations(branches, r=2):
            if 1 + d1 + d2 > maxinnerdepth:
                maxinnerpath    =  b1 + [ v ] + b2 
                maxinnerdepth   = 1 + d1 + d2        

        for b,d in branches:
            if 1 + d > maxdepth:
                maxpath = [v] + b
                maxdepth = 1 + d
        return ((maxpath,maxdepth), (maxinnerpath, maxinnerdepth)) 
    
    ((maxpath,maxdepth), (maxinnerpath, maxinnerdepth)) = DFS_tree_depth(Tree, list(Tree.nodes)[0])
    print(maxdepth, maxinnerdepth)
    return maxpath if maxdepth > maxinnerdepth else maxinnerpath



def alternate_path_v2(G : nx.Graph, terms, permutations):
    
    T, _ = sample_path(G, terms)
    Q = get_Diameter(T)
    other_color = set()

    ret = []
    # print(Q)
    # print(list(G.nodes()))
    # # exit(0)
    made_progress = True
    while len(Q) > 2 and made_progress:
        made_progress = False
        for (u,v) in Q:
            for H in [u,v]:
                # print(H)
                if H.parent not in other_color:
                    ret.append(H)
                    other_color.add(H.parent)
                for w in list(G.nodes()):
                    if H in w:
                        G.remove_node(w)
                        made_progress = True
                    # print(w)
        T, _ = sample_path(G, terms)
        if T.number_of_edges() > 1:
            Q = get_Diameter(T)
        else:
            break
        # Q = [ ]

    last_base = None
    print(T.number_of_edges())
    for e in T.edges:
        ((v1,v2),(u1,u2)) = e
        # if last_base != None and T.edges[e]['permutation']:
        #     print("W"*20)
        #     for k in [ last_base, T.edges[e]['permutation'] ]:
        #         ret.append( SWAPlocal_Hamiltonian(permutations[k].arr) )
        for v in [ v1, v2, u1, u2]:
            if v.parent not in other_color:
                ret.append(v)
                other_color.add(v.parent)
    #     if last_base == None:
    #         ret.append( SWAPlocal_Hamiltonian(\
    #             permutations[T.edges[e]['permutation']].arr) )
    #     last_base = T.edges[e]['permutation']
    # if last_base != None:
    #     ret.append( SWAPlocal_Hamiltonian(permutations[last_base].arr) )
    for term in terms:
        if term.parent not in other_color:
            ret.append(term)

    return ret, terms
    #     print(e)

    
if __name__ == "__main__":
    # alternate_path_v2()


    # circuit = QuantumCircuit(10)
    G, terms, permus = generated_the_product_graph(num_of_terms=40) 

    # pos = nx.spring_layout(G, seed=50)

    # nx.draw_networkx_nodes(G, pos)
    # nx.draw_networkx_edges(G, pos)
    # plt.show()

    for term in sorted(terms, key=lambda x : "".join(x.tensor)):
        print("".join(term.tensor))
    canidates = [ ]
    for _ in range(10):    
        path, terms = alternate_path_v2(deepcopy(G), terms, permus)
        circuit = genreate_circut(path)
        depth = genreate_optimzed_circut(circuit, terms)
        canidates.append( (depth, circuit) )

    depth, circuit =  min( canidates, key = lambda x : x[0] )
    genreate_optimzed_circut(circuit, terms, svg = False, entire=True)
    # terms = shuffle(terms)
    # T = nx.Graph()
    # for term in terms:
    #     if term not in color:
    #         color.add(term)
    #         select filter(lambda x : (x not in color) and (len(G.adj[(term, x)]) > 0) , terms):
    #             if G.adj[(term,  )]