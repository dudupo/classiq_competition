
from Hamiltonian_parser import PATHFILE

def majority_base():
    _str = open(PATHFILE).read()
    X,Y,Z,I =  (_str.count(c) for c in ["I", "X", "Z", "Y"])
    print( f"stats:\n\tI:{I}\n\tX : {X}\n\tY : {Y}\n\tZ: {Z}")

if __name__ == "__main__":
    majority_base() 