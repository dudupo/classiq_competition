
from importlib.resources import path
from math import ceil
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.visualization import circuit_drawer
from matplotlib import pyplot as plt
from random import randint
import datetime

from requests import delete

'''
    Given, P is pauli operetor => P^2 = I.  
    e^(iwP) = cos(w)I +  sin(w)P
'''
"+  0.003034656830204855 * IIIXXIIIYY"


WIRES = 10
TEST  = "NOT" #"PRODUCT"
PATHFILE =  { 
    "NOT" : "./LiH",
    "YES" : "./LiH_test",
    "PRODUCT" : "./LiH_test_product_space" 
    }[TEST]
RANDOMSTACTICS = True
STRATEGY =  "PRODUCT" #"HIURISTIC" #,"RANDOM" ,"NON"

# the number of products (N plays the role of n-> \infity)
N = 1


class local_Hamiltonian():
    def __init__(self, tensor, weight) -> None:
        self.tensor = tensor
        self.weight = weight
    
    def tensorspace(self, other) -> bool:
        for A,B in zip( list(self.tensor), list(other.tensor)):
            # print(A,B)
            if "I" not in [A, B]:
                return False 
        return True

def parser_line(line) -> local_Hamiltonian:
    line  = line.split()
    return local_Hamiltonian( list(line[-1]),
     { "-" : -1 , "+" : 1 }[line[0]] * np.float64(line[1]) / N )

def parser() -> None:
    hamiltonians = [ ] 
    for line in open(PATHFILE).readlines():
        if len(line) > 1:
            hamiltonians.append(parser_line(line))
    return hamiltonians

def donothing(_):
    pass


class CirGraph():
    def __init__(self, wires) -> None:
        self.wires = wires
        self.wires_stacks = [ [] for _ in range(wires) ]  
        self.hist = 0 

    # def pop(self, wire, index, circuit):
    #     GRAPH.wires_stacks.pop(-1)
    #     circuit.data.pop(index - self.hist)
    # def append(self, wire, )


GRAPH = CirGraph(WIRES) 

def rotateY(cir):
    def _func(wire):
        cir.s(wire)
        cir.h(wire)
    return _func

def unrotateY(cir):
    def _func(wire):
        cir.h(wire)
        cir.sdg(wire)
    return _func

def MulByterm(circuit : QuantumCircuit, term, main_wire = WIRES-1, sign=1) -> QuantumCircuit:
        
    def reqursive_manner(tensor, wire, weight):
        
        if wire == main_wire:
            # little angle approximation. 
            circuit.rz(2*weight, main_wire)
            return 
        
        pauli = tensor[wire]
        
        compute =   {   
            "X" : lambda cir : cir.h,    
            "Y" : lambda cir : rotateY(cir),
            "Z" : lambda cir : donothing,
            "I" : lambda cir : donothing }
        

        uncompute = {
            "X" : lambda cir : cir.h,   
            "Y" : lambda cir : unrotateY(cir),
            "Z" : lambda cir : donothing,
            "I" : lambda cir : donothing  }
        
        
        if pauli != "I":
            compute[pauli](circuit)(wire)
            circuit.cx(wire, main_wire)
        
        reqursive_manner(tensor, (wire + sign) % WIRES, weight)
        
        if pauli != "I":
            circuit.cx(wire, main_wire)
            uncompute[pauli](circuit)(wire)
         
    
    reqursive_manner(term.tensor, (main_wire + sign) % WIRES, term.weight)
    return circuit


def cutting(circuit : QuantumCircuit):
    '''Second optimization, cuts the gates which are follwed by their
    uncompute '''
    def filter_by_wire(wire):
        return list(filter( lambda item :\
            any( [register.index == wire for register in item[1][1]] ), enumerate(circuit.data) ))
    

    UNCOMPTE = { 
        "h"  : "h",
        "cx" : "?",
        "rz" : "?" ,
        "s"  : "sdg",
        "sdg" : "s" }

    

    indices_todelete = []

    for wire in range(WIRES):
        operators = filter_by_wire(wire)

        j = 0
        while (j < len(operators) - 1 ):
            if ( UNCOMPTE[operators[j][1][0].name] == operators[j+1][1][0].name ):
                indices_todelete.append(  operators[j][0]  )
                indices_todelete.append(  operators[j+1][0]  )
            j += 1 
    
    for index in reversed(sorted(indices_todelete)):
        circuit.data.pop(index)
    
    return circuit


def alternate(i):
    return 0 if i % 2 == 0 else 9
    # return 4*i % WIRES #{0: 0 , 1 : 5, 2:9 }[i%3]  

def support_edge(loacl_hamiltonain : local_Hamiltonian, reverse = False):
    
    operators = enumerate(list(loacl_hamiltonain.tensor)) if not reverse else \
        reversed(list(enumerate(list(loacl_hamiltonain.tensor))))

    for j,op in operators:
        if op != "I":
            return j
    
    return 0 if not reverse else WIRES-1

def genreate_circut(terms = None):
    circuit = QuantumCircuit(10)
    terms = parser() if terms == None else terms
    
    
    for i, term in enumerate(terms):
        
        sign =   { 
                0 : 1,
                1 : -1,
        }[i%2]
    
        reverse = { 
                0 : True,
                1 : False,
            }[i % 2]

        if "STRATEGY" == "PRODUCT":
            sign =   { 
                    0 : 1,
                    1 : 1,
                    2 : -1,
                    3 : -1
                }[i % 4]  #if "STRATEGY" == "HIURISTIC" else 1

            reverse = { 
                    0 : True,
                    1 : True,
                    2 : False,
                    3 : False
                }[i % 4]

        main_wire = {
            "NON"       : lambda :  WIRES - 1,
            "RANDOM"    : lambda :  randint(0, WIRES - 1),
            "HIURISTIC" : lambda : support_edge(term, reverse=reverse),
            "PRODUCT"   : lambda : support_edge(term, reverse=reverse)
        }[STRATEGY]()

        MulByterm(circuit, term, main_wire, sign=sign) 

    return  circuit

def genreate_optimzed_circut(circuit, terms):
    # circuit = genreate_circut(terms)
    circuit = cutting(cutting(circuit))
    
    for _ in range(ceil(np.log(N))):
        circuit = circuit.compose(circuit)
    circuit = cutting(circuit)

    circuit_drawer(circuit, output='mpl',style="bw", fold=-1)
    plt.title( f"TERMS: {len(terms)}, DEPTH:{circuit.depth()}")
    plt.savefig(f'Ham_{STRATEGY}-{datetime.datetime.now()}.svg')
    open(f"Ham_{STRATEGY}-{datetime.datetime.now()}.qasm", "w+").write(circuit.qasm())
    print(f"TERMS: {len(terms)}, DEPTH:{circuit.depth()}")



class Node():
    def __init__(self, local_hamiltonian : local_Hamiltonian) -> None:
        self.local_hamiltonian = local_hamiltonian
        self.edges = [ ]



def generateBiparatite(Graph, l="XXXXXIIIII", r ="IIIIIXXXXX"):
    ghost_l = local_Hamiltonian(l, 0)
    ghost_r = local_Hamiltonian(r, 0)
    L, R = [], []
    for v in Graph:
        if v.local_hamiltonian.tensorspace(ghost_l):
            L.append(v)
        elif v.local_hamiltonian.tensorspace(ghost_r):
            R.append(v)
    
    Path = []
    for u,v in zip(L,R):
        Path.append((u,v))
        # Path.append(v.local_hamiltonian)
    print(len(Path))
    return Path

def generateGraph(hamiltonians):
    vertcies = [ Node(hamiltonian) for hamiltonian in hamiltonians ]
    edges = 0
    for j, u in enumerate(vertcies):
        for v in vertcies[j+1:]:
            if u.local_hamiltonian.tensorspace(v.local_hamiltonian):
                v.edges.append(u)
                u.edges.append(v)
                edges += 1
    print(f"edges: {edges}")
    return vertcies

def DFS(Graph):
    color = set()
    
    alternate_path = []
    def _DFS(v : Node, _color : set, parity, path = []):
        for u in v.edges:
            if u not in _color:
                _color.add(u)
                if parity == 0:
                    path.append( (v,u) )
                    _DFS(u, _color, 1, path)    
                    parity = 1
                else:
                    _DFS(u, _color, 0, path)
        return path
    
    for v in Graph:
        if v not in color:
            color.add(v)
            alternate_path += _DFS(v, color, 0, [])
    # print(len(alternate_path))
    return alternate_path

def deleteNode(v : Node):
    for u in v.edges:
        u.edges.remove(v)
        


def delete_path(Graph, path):
    for u,v in path:
        deleteNode(u)
        deleteNode(v)
        Graph.remove(u)
        Graph.remove(v)
    

def decompiseToAlternatePath(Graph):
    alternate_paths = [ ]

    for i in range(0, 5):
        for j in range(1,WIRES-1):
            path = generateBiparatite(Graph,\
                    l="X"*(j-i) + "I"*(WIRES-j+i), r ="I"*(j+i) + "X"*(WIRES-j-i))

            delete_path(Graph, path)
            for u,v in path:
                alternate_paths.append(u.local_hamiltonian)
                alternate_paths.append(v.local_hamiltonian)   
    

    path = DFS(Graph)

    while len(path) > 0: 
        delete_path(Graph, path)
        for u,v in path:
            alternate_paths.append(u.local_hamiltonian)
            alternate_paths.append(v.local_hamiltonian)            
        path = DFS(Graph)


    normalpath = []
    for v in Graph:
        normalpath.append(v.local_hamiltonian)            
    return alternate_paths, normalpath

if __name__  == "__main__":
    # genreate_circut()
    from random import sample
    terms = sample(parser(), 30)
    alternate_paths, normalpath = decompiseToAlternatePath(generateGraph(terms))
    print(f"length of alternate_paths {len(alternate_paths)}")
    STRATEGY = "PRODUCT"
    circuit_product = genreate_circut(alternate_paths)
    STRATEGY = "HIURISTIC"
    circuit_steps   = genreate_circut(normalpath)
    STRATEGY = "PRODUCT|HIURISTIC"
    genreate_optimzed_circut(circuit_product.compose(circuit_steps),\
         alternate_paths + normalpath )