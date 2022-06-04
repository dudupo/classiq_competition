
from importlib.resources import path
from math import ceil
from tkinter.tix import Tree
from qiskit import  QuantumCircuit
from qiskit.visualization import circuit_drawer
from matplotlib import pyplot as plt
import datetime

N                       = 4
WIRES                   = 10
TEST                    = "NOT" 
RANDOMSTACTICS          = True
STRATEGY                =  "PRODUCT" 
OUTSIDE_OF_CIRCUIT      = WIRES + 1
PATHFILE                =  { 
                            "NOT" : "./LiH",
                            "YES" : "./LiH_test",
                            "PRODUCT" : "./LiH_test_product_space" 
                            }[TEST]


class local_Hamiltonian():
    def __init__(self, tensor, weight) -> None:
        self.tensor = tensor
        self.weight = weight
        self.parent = self
    
    def tensorspace(self, other) -> bool:
        for A,B in zip( list(self.tensor), list(other.tensor)):
            if "I" not in [A, B]:
                return False 
        return True

    def dis(self, other) -> int:
        ret = 0
        for A,B in zip(list(self.tensor), list(other.tensor)):
            if A != B:
                ret += 1
        return ret
    
    def solid_product(self, other):
        indices = []
        for j in range(1,WIRES):
            l= "X" * j + "I" * (WIRES-j)
            r = "I" * j + "X" * (WIRES-j)
            if local_Hamiltonian(l,0).tensorspace(self) and\
                local_Hamiltonian(r,0).tensorspace(other):
                indices.append(j)
        
        if len(indices) > 0:
            return True, indices 
        return False, []
    
    def newbase(self, perm):
        tensor = [ "" ] * len(self.tensor)
        for i in range(len(perm)):
            tensor[i] = self.tensor[perm[i]]
        ret = local_Hamiltonian(tensor,self.weight)
        ret.parent = self.parent
        return ret
    
    def median(self):
        support = list(filter( lambda x : self.tensor[x] != 'I', range(WIRES)))
        if len(support) != 0:
            return support[int(len(support)/2)] 
        return 0

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

def MulByterm (circuit : QuantumCircuit, term ,next_terms = [], last_terms =  [],
 main_wire = WIRES-1) -> QuantumCircuit:
        
    def reqursive_manner(tensor, wire, weight, last_wire, _sign, first_not_trival=True):
        
        if wire < 0 or wire == WIRES:
            return QuantumCircuit(WIRES),QuantumCircuit(WIRES)

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
        
        pauli = tensor[wire]
        if wire == main_wire:
            circuit_node = QuantumCircuit(WIRES)
            LU, RU = reqursive_manner(tensor, wire-1, weight, last_wire, -1, first_not_trival = True)
            LD, RD = reqursive_manner(tensor, wire+1, weight, last_wire,  1, first_not_trival = True)
            
            compute[pauli](circuit_node)(wire)
            for L in [ LD, LU]:
                circuit_node = circuit_node.compose(L)
            circuit_node.rz(2*weight, main_wire)
            for R in [ RD, RU]:
                circuit_node = circuit_node.compose(R)
            uncompute[pauli](circuit_node)(wire)  
            return  circuit_node
        
        if pauli == 'I':
            
            
            temp_wire = last_wire + _sign if first_not_trival else last_wire
            return reqursive_manner(tensor, wire + _sign, weight,
             last_wire, _sign, first_not_trival = True)
            
        else:
            parity_collector = False
            if first_not_trival:
                parity_collector = True
 
            L, R = reqursive_manner(tensor, wire + _sign, weight, main_wire + _sign , _sign, first_not_trival = False)
            circuit_left, circuit_right = QuantumCircuit(WIRES),QuantumCircuit(WIRES)
            
            if (parity_collector) or not (\
                 (last_terms[wire][0] == pauli) and\
                  (last_terms[wire][1] == main_wire)):

            
                compute[pauli](circuit_left)(wire)
                circuit_left = circuit_left.compose(L)
                circuit_left.cx(wire, last_wire)
            
            else:
                circuit_left = L 

            if  (parity_collector) or not (
                 (next_terms[wire][0] == pauli) and\
                  (next_terms[wire][1] == main_wire)):
            
                circuit_right.cx(wire, last_wire)
                circuit_right = circuit_right.compose(R)

                uncompute[pauli](circuit_right)(wire)
            else:                
                circuit_right = R
            return circuit_left, circuit_right

    circuit = circuit.compose(reqursive_manner(term.tensor, main_wire, term.weight, main_wire, 0))
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
        "sdg" : "s",
        "sxdg" :"?" }

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


def genreate_circut(terms = None):
    circuit = QuantumCircuit(WIRES)
    terms = parser() if terms == None else terms
    print(len(terms))
    for i, term in enumerate(terms):
        
        main_wire =  term.median() 
        next_terms,last_terms = [],[]

        for _j in range(WIRES):
            found = False
            if i+1 < len(terms):
                for _term in terms[i+1:]: 
                    if _term.tensor[_j] != 'I':
                        next_terms.append( (_term.tensor[_j], _term.median() ))
                        found = True     
                        break 
            if not found:
                next_terms.append( ('I', OUTSIDE_OF_CIRCUIT ))
            
            found = False
            if i > 0:
                for _term in reversed(terms[i-1:]): 
                    if _term.tensor[_j] != 'I':
                        last_terms.append( (_term.tensor[_j], _term.median() ))
                        found = True
                        break      
            if not found:
                last_terms.append( ('I', OUTSIDE_OF_CIRCUIT  ))

        circuit = MulByterm(circuit, term, main_wire=main_wire,
         next_terms=next_terms, last_terms=last_terms ) 

    return  circuit

def genreate_optimzed_circut(circuit, terms, svg =False, entire = False):
    circuit = cutting(cutting(circuit))
    
    if entire:
        for _ in range(ceil(np.log(N))):
            circuit = circuit.compose(circuit)
        circuit = cutting(circuit)

    print(f"TERMS: {len(terms)}, DEPTH:{circuit.depth()}")
    
    if svg:
        circuit_drawer(circuit, output='mpl',style="bw", fold=-1)
        plt.title( f"TERMS: {len(terms)}, DEPTH:{circuit.depth()}")
        plt.savefig(f'Ham_{STRATEGY}-{datetime.datetime.now()}.svg')
    
    if entire:
        open(f"Ham_{STRATEGY}-{datetime.datetime.now()}.qasm", "w+").write(circuit.qasm())
    
    return circuit.depth()




if __name__  == "__main__":
    path = [ local_Hamiltonian( "XIXZZXIXXZ", 0.5 ),  local_Hamiltonian( "XIXXZXIXXZ", 0.5 ) ] 
    circuit =  genreate_circut(path)
    genreate_optimzed_circut(circuit ,path, svg=True, entire=False)