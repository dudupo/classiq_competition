
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.visualization import circuit_drawer
from matplotlib import pyplot as plt

'''
    Given, P is pauli operetor => P^2 = I.  
    e^(iwP) = cos(w)I +  sin(w)P
'''

"+  0.003034656830204855 * IIIXXIIIYY"


WIRES = 10

class local_Hamiltonian():
    def __init__(self, tensor, weight) -> None:
        self.tensor = tensor
        self.weight = weight

def parser_line(line) -> local_Hamiltonian:
    line  = line.split()
    return local_Hamiltonian( line[-1],
     { "-" : -1 , "+" : 1 }[line[0]] * np.float64(line[1]) )

def parser() -> None:
    hamiltonians = [ parser_line(line)\
         for line in open("./LiH").readlines()]
    
def MulByterm(circuit : QuantumCircuit, term, main_wire = WIRES-1) -> QuantumCircuit:
    
    
    def reqursive_manner(tensor, wire, weight):
        
        if wire == main_wire:
            # little angle approximation. 
            circuit.rz(2*weight, main_wire)
            return 
        
        puli = tensor.pop(0)
        
        compute =   {   
            "X" : lambda cir : cir.H,    
            "Y" : lambda cir : cir.H,
            "Z" : lambda cir : cir.H,
            "I" : lambda cir : (lambda _ : pass )}
        
        uncompute = {
            "X" : lambda cir : cir.H,   
            "Y" : lambda cir : cir.H,
            "Z" : lambda cir : cir.H,
            "I" : lambda cir : (lambda _ : pass)  }
        
        compute[puli](wire)
        circuit.cx(main_wire, wire)
        reqursive_manner(tensor)
        circuit.cx(main_wire, wire)
        uncompute[puli](wire)
         
    
    reqursive_manner(term.tensor, 0, term.weight)
    return circuit

def genreate_circut():
    circuit = QuantumCircuit(10)
    terms = parser()
    for term in terms:
        MulByterm(circuit, term, WIRES-1) 
    
    circuit_drawer(circuit, output='mpl',style="bw", fold=-1)
    plt.savefig('Ham.svg')
    plt.show()       
