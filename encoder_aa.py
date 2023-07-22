import os
import numpy as np
from icecream import ic

gstr = 'GAVLIMPWFYSTNQCRDEHK'

def QWE_AA(word):
    char = list(word)
    # INITIALIZE PLACEHOLDER FOR TENSORS
    l0 = [] #P1
    l1 = [] #P2
    l2 = [] #P3
    l3 = [] #P4
    xi_0 = []
    xi_1 = []
    xi_2 = []
    xi_3 = []
    print(char)
    for c in char:
        if (c in ['G','A','V','L','I','M','P']) or (c in ['g','a','v','l','i','m','p']):
            l0.append(1)
            l1.append(0)
            l2.append(0)
            l3.append(0)
            print(c)
            if (c == 'G') or (c == 'g'):
                print()
                xi_0.append(np.exp(2*np.pi*1j*0/7))
                xi_1.append(0)
                xi_2.append(0)
                xi_3.append(0)
            elif (c == 'A') or (c == 'a'):
                xi_0.append(np.exp(2*np.pi*1j*1/7))
                xi_1.append(0)
                xi_2.append(0)
                xi_3.append(0)
            elif (c == 'V') or (c == 'v'):
                xi_0.append(np.exp(2*np.pi*1j*2/7))
                xi_1.append(0)
                xi_2.append(0)
                xi_3.append(0)
            elif (c == 'L') or (c == 'l'):
                xi_0.append(np.exp(2*np.pi*1j*3/7))
                xi_1.append(0)
                xi_2.append(0)
                xi_3.append(0)
            elif (c == 'I') or (c == 'i'):
                xi_0.append(np.exp(2*np.pi*1j*4/7))
                xi_1.append(0)
                xi_2.append(0)
                xi_3.append(0)
            elif (c == 'M') or (c == 'm'):
                xi_0.append(np.exp(2*np.pi*1j*5/7))
                xi_1.append(0)
                xi_2.append(0)
                xi_3.append(0)
            elif (c == 'P') or (c == 'p'):
                xi_0.append(np.exp(2*np.pi*1j*6/7))
                xi_1.append(0)
                xi_2.append(0)
                xi_3.append(0)
        elif (c in ['W','F','Y']) or (c in ['w','f','y']):
            l1.append(1)
            l0.append(0)
            l2.append(0)
            l3.append(0)
            if (c == 'W') or (c == 'w'):
                xi_1.append(np.exp(2*np.pi*1j*0/3))
                xi_0.append(0)
                xi_2.append(0)
                xi_3.append(0)
            elif (c == 'F') or (c == 'f'):
                xi_1.append(np.exp(2*np.pi*1j*1/3))
                xi_0.append(0)
                xi_2.append(0)
                xi_3.append(0)
            elif (c == 'Y') or (c == 'y'):
                xi_1.append(np.exp(2*np.pi*1j*2/3))
                xi_0.append(0)
                xi_2.append(0)
                xi_3.append(0)
        elif (c in ['S','T','N','Q','C']) or (c in ['s','t','n','q','c']):
            l2.append(1)
            l0.append(0)
            l1.append(0)
            l3.append(0)
            if (c == 'S') or (c == 's'):
                xi_2.append(np.exp(2*np.pi*1j*0/5))
                xi_0.append(0)
                xi_1.append(0)
                xi_3.append(0)
            elif (c == 'T') or (c == 't'):
                xi_2.append(np.exp(2*np.pi*1j*1/5))
                xi_0.append(0)
                xi_1.append(0)
                xi_3.append(0)
            elif (c == 'N') or (c == 'n'):
                xi_2.append(np.exp(2*np.pi*1j*2/5))
                xi_0.append(0)
                xi_1.append(0)
                xi_3.append(0)
            elif (c == 'Q') or (c == 'q'):
                xi_2.append(np.exp(2*np.pi*1j*3/5))
                xi_0.append(0)
                xi_1.append(0)
                xi_3.append(0)
            elif (c == 'C') or (c == 'c'):
                xi_2.append(np.exp(2*np.pi*1j*4/5))
                xi_0.append(0)
                xi_1.append(0)
                xi_3.append(0)
        elif (c in ['R','D','E','H','K']) or (c in ['r','d','e','h','k']):
            l3.append(1)
            l0.append(0)
            l1.append(0)
            l2.append(0)
            if (c == 'R') or (c == 'r'):
                xi_3.append(np.exp(2*np.pi*1j*0/5))
                xi_0.append(0)
                xi_1.append(0)
                xi_2.append(0)
            elif (c == 'D') or (c == 'd'):
                xi_3.append(np.exp(2*np.pi*1j*1/5))
                xi_0.append(0)
                xi_1.append(0)
                xi_2.append(0)
            elif (c == 'E') or (c == 'e'):
                xi_3.append(np.exp(2*np.pi*1j*2/5))
                xi_0.append(0)
                xi_1.append(0)
                xi_2.append(0)
            elif (c == 'H') or (c == 'h'):
                xi_3.append(np.exp(2*np.pi*1j*3/5))
                xi_0.append(0)
                xi_1.append(0)
                xi_2.append(0)
            elif (c == 'K') or (c == 'k'):
                xi_3.append(np.exp(2*np.pi*1j*4/5))
                xi_0.append(0)
                xi_1.append(0)
                xi_2.append(0)

    # POSTPROCESSING: CONCATENATE (STACKING)
    l0 = np.array(l0).reshape(1,-1)
    l1 = np.array(l1).reshape(1,-1)
    l2 = np.array(l2).reshape(1,-1)
    l3 = np.array(l3).reshape(1,-1)
    xi_0 = np.array(xi_0).reshape(1,-1)
    xi_1 = np.array(xi_1).reshape(1,-1)
    xi_2 = np.array(xi_2).reshape(1,-1)
    xi_3 = np.array(xi_3).reshape(1,-1)

    ic(l0,l1,l2,l3)
    ic(xi_0,xi_1,xi_2,xi_3)
    PE = np.concatenate([l0,l1,l2,l3], axis = 0)
    QE = np.concatenate([xi_0,xi_1,xi_2,xi_3], axis = 0)
    # RETURN [POSISION ENCODING, QUANTUM ENCODING]
    return PE, QE




                

def ComputeQuantumEmbeddings(x):
    xT = np.conj(x).T
    ic(x.shape)
    ic(xT.shape)
    return np.matmul(xT ,x)
