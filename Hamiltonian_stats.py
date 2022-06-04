
from Hamiltonian_parser import PATHFILE
import numpy as np 
from matplotlib import pyplot as plt
import pickle as pkl
import datetime
from Hamiltonian_parser import parser

def majority_base():
    _str = open(PATHFILE).read()
    X,Y,Z,I =  (_str.count(c) for c in ["I", "X", "Z", "Y"])
    print( f"stats:\n\tI:{I}, {I/(X+Y+Z+I)} \n\tX : {X}, {X/(X+Y+Z+I)}\n\tY : {Y}, {Y/(X+Y+Z+I)}\n\tZ: {Z}, {Z/(X+Y+Z+I)}")

def _majority_base():
    terms = parser()

    def support(number, term):
        for x,y in zip(format(number, '#010b')[2:], term):
            if x == '0' and y != 'X':
                return 0 
        return 1

    count = np.zeros(2**10)
    for i in range(2**10):
        for term in terms:
            count[i] += support(i, term.tensor)
    
    pkl.dump( count, open(f"histogram_pickle-{datetime.datetime.now()}.pkl" , "wb+"))
    plt.bar(list(range(2**10)), count)
    plt.show()
    
    print(  min([ "".join(term.tensor) for term in terms], key = lambda x : x.count("X")))




if __name__ == "__main__":
    majority_base() 