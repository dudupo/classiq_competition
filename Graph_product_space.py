import networkx as nx
from numpy import argmin
from sympy import permutedims
from Hamiltonian_parser import parser, local_Hamiltonian,genreate_circut,genreate_optimzed_circut,SWAPlocal_Hamiltonian
from itertools import permutations, product

from random import sample, shuffle
from  tqdm import tqdm
import pickle as pkl

class Permutation_Base():
    def __init__(self, arr) -> None:
        self.arr = arr
        self.parent = self
def generated_the_product_graph(num_of_terms=20):
    terms = parser()# sample(parser(), num_of_terms)
    def generated_the_product_graph_by_base(_terms):
        
        G = nx.Graph()

        def check_solid_edge(H1, H2, H3, H4):
            res = [Hi.solid_product(Hj) for Hi,Hj in [ (H1,H2), (H1,H4), (H2, H3), (H3, H4)]]
            return all([x[0] for x in res])

        def check_edge(H1, H2, H3, H4):
            if H1.dis(H2) <= 1 and H1.dis(H4) <= 1 and  H2.dis(H3) <= 1 and H3.dis(H4) <= 1:
                return True, max(H1.dis(H3), H2.dis(H4))
            else :
                return False, None 
        
        for (H1, H2) in product(_terms, _terms):
            G.add_node((H1, H2))

        for (_,((H1, H2), (H3, H4))) in enumerate( product(\
            product(_terms,_terms),product(_terms,_terms))):
            print(len(_terms)**4 -  _)
            exist, cost = check_edge(H1, H2, H3, H4)
            if exist:
                G.add_edge( (H1, H2), (H3, H4) )
                G.edges[(H1, H2), (H3, H4)]['weight'] = cost
                if check_solid_edge(H1, H2, H3, H4):
                    G.edges[(H1, H2), (H3, H4)]['solid'] = True 
                else:
                    G.edges[(H1, H2), (H3, H4)]['solid'] = False
        return G, _terms 
    


            

    permutations = list(map(lambda x: Permutation_Base(x) , [
        [0,2,4,6,8,1,3,5,7,9],
        [0,1,2,3,4,5,6,7,8,9]
    ]))
    graphs = []
    perm_terms = []
    mainG = nx.Graph()
    for j, permutation in enumerate(permutations): 
        perm_terms = list(map( lambda x : x.newbase(permutation.arr), terms))
        # print(perm_terms)
        G, _ = generated_the_product_graph_by_base(perm_terms) 
        for e in G.edges:
            G.edges[e]["permutation"] = j
        
        # graphs.append(G)
        mainG = nx.compose(mainG, G)
            
    #     # print(len(mainG.nodes))

    # for l, ((G1,p1), (G2,p2)) in enumerate(product(zip(graphs,permutations)\
    #      ,zip(graphs,permutations))):
    #     if G1 != G2:
    #         G.add_node( (p1,p2) )
    #         G.add_edges_from([(node,(p1,p2), {'weight': 1}) for node in G1.nodes])
    #         G.add_edges_from([((p1,p2),node,  {'weight': 0}) for node in G2.nodes])
    # mainG, _ = generated_the_product_graph_by_base(mainG)
    
    # for e in mainG.edges:
    #     print(mainG.edges[e]["permutation"])    
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
def sample_path(G, terms):
    color = set()
    
    def DFS(v, T, _color, flag =True):        
        term1, term2 = v
        
        if (term1.parent in _color) or (term2.parent in _color):
            return  _color
        
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
                _color = DFS(u, T, _color, flag=flag)
        return _color
    T = nx.Graph()
    l = []

    for v in G.nodes():
        T.add_node(v)
        l.append(v)
    
    
    last_size = len(T.edges.values())
    for v in G.nodes(): 
        color = DFS(v, T, color, flag=False)
        if last_size == len(T.edges.values()):
            for u in [v[0], v[1]]:
                if u.parent in color:
                    color.remove(u.parent)

        last_size = len(T.edges.values())
    for v in G.nodes(): 
        color = DFS(v, T, color, flag=True)

    return T, color

def alternate_path_v2(G, terms, permutations):
    T, color = sample_path(G, terms)
    ret = []
    other_color = set()
    last_base = None
    for e in T.edges:
        ((v1,v2),(u1,u2)) = e
        if last_base != None and T.edges[e]['permutation'] != last_base:
            print("W"*20)
            for k in [ last_base, T.edges[e]['permutation'] ]:
                ret.append( SWAPlocal_Hamiltonian(permutations[k].arr) )
        for v in [ v1, v2, u1, u2]:
            if v.parent not in other_color:
                ret.append(v)
                other_color.add(v.parent)
        if last_base == None:
            ret.append( SWAPlocal_Hamiltonian(\
                permutations[T.edges[e]['permutation']].arr) )
        last_base = T.edges[e]['permutation']

    ret.append( SWAPlocal_Hamiltonian(permutations[last_base].arr) )
    for term in terms:
        if term.parent not in other_color:
            ret.append(term)

    return ret, terms
    #     print(e)


if __name__ == "__main__":
    # alternate_path_v2()


    # circuit = QuantumCircuit(10)
    G, terms, permus = generated_the_product_graph(num_of_terms=20) 
    for term in sorted(terms, key=lambda x : "".join(x.tensor)):
        print("".join(term.tensor))
    canidates = [ ]
    for _ in range(50):    
        path, terms = alternate_path_v2(G, terms, permus)
        circuit = genreate_circut(path)
        depth = genreate_optimzed_circut(circuit, terms)
        canidates.append( (depth, circuit) )

    depth, circuit =  min( canidates, key = lambda x : x[0] )
    genreate_optimzed_circut(circuit, terms, svg = True)
    # terms = shuffle(terms)
    # T = nx.Graph()
    # for term in terms:
    #     if term not in color:
    #         color.add(term)
    #         select filter(lambda x : (x not in color) and (len(G.adj[(term, x)]) > 0) , terms):
    #             if G.adj[(term,  )]