import pandas as pd
import numpy as np
import os
from icecream import ic

def tensorize(word):
    l0 = [] #Q_vowel 1
    l1 = [] #Q_consonant 1
    l2 = [] #Q_consonant 2
    l3 = [] #Q_consonant 3
    char = list(word)
    xi_0 = []
    xi_1 = []
    xi_2 = []
    xi_3 = []
    for c in char:
        if (c in ['a', 'e','i', 'u', 'o']) or (c in ['A', 'E', 'I', 'U', 'O']):
            # BINARY ENCODING VOWEL-CONSONANT
            l0.append(1)
            l1.append(0)
            l2.append(0)
            l3.append(0)
            # XI ENCODING
            if (c == 'a') or (c == 'A'):
                xi_0.append(np.exp(2*np.pi*1j*0/5))
                xi_1.append(0)
                xi_2.append(0)
                xi_3.append(0)

            elif (c == 'e') or (c == 'E'):
                xi_0.append(np.exp(2*np.pi*1j*1/5))
                xi_1.append(0)
                xi_2.append(0)
                xi_3.append(0)
            elif (c == 'i') or (c == 'I'):
                xi_0.append(np.exp(2*np.pi*1j*2/5))
                xi_1.append(0)
                xi_2.append(0)
                xi_3.append(0)
            elif (c == 'u') or (c == 'U'):
                xi_0.append(np.exp(2*np.pi*1j*3/5))
                xi_1.append(0)
                xi_2.append(0)
                xi_3.append(0)
            elif (c == 'o') or (c == 'O'):
                xi_0.append(np.exp(2*np.pi*1j*4/5))
                xi_1.append(0)
                xi_2.append(0)
                xi_3.append(0)

        elif (c in ['b', 'c','d', 'f', 'g', 'h', 'j']) or (c in ['B', 'C', 'D', 'F', 'G', 'H', 'J']): 
            l0.append(0)
            l1.append(1)
            l2.append(0)
            l3.append(0)
            # XI ENCODING
            if (c == 'b') or (c == 'B'):
                xi_1.append(np.exp(2*np.pi*1j*0/7))
                xi_0.append(0)
                xi_2.append(0)
                xi_3.append(0)
            elif (c == 'c') or (c == 'C'):
                xi_1.append(np.exp(2*np.pi*1j*1/7))
                xi_0.append(0)
                xi_2.append(0)
                xi_3.append(0)
            elif (c == 'd') or (c == 'D'):
                xi_1.append(np.exp(2*np.pi*1j*2/7))
                xi_0.append(0)
                xi_2.append(0)
                xi_3.append(0)
            elif (c == 'f') or (c == 'F'):
                xi_1.append(np.exp(2*np.pi*1j*3/7))
                xi_0.append(0)
                xi_2.append(0)
                xi_3.append(0)
            elif (c == 'g') or (c == 'G'):
                xi_1.append(np.exp(2*np.pi*1j*4/7))
                xi_0.append(0)
                xi_2.append(0)
                xi_3.append(0)
            elif (c == 'h') or (c == 'H'):
                xi_1.append(np.exp(2*np.pi*1j*5/7))
                xi_0.append(0)
                xi_2.append(0)
                xi_3.append(0)
            elif (c == 'j') or (c == 'J'):
                xi_1.append(np.exp(2*np.pi*1j*6/7))
                xi_0.append(0)
                xi_2.append(0)
                xi_3.append(0)

        elif (c in ['k', 'l','m', 'n', 'p', 'q', 'r']) or (c in ['K', 'L', 'M', 'N', 'P', 'Q', 'R']): 
            l0.append(0)
            l1.append(0)
            l2.append(1)
            l3.append(0)
            # XI ENCODING
            if (c == 'k') or (c == 'K'):
                xi_2.append(np.exp(2*np.pi*1j*0/7))
                xi_0.append(0)
                xi_1.append(0)
                xi_3.append(0)
            elif (c == 'l') or (c == 'L'):
                xi_2.append(np.exp(2*np.pi*1j*1/7))
                xi_0.append(0)
                xi_1.append(0)
                xi_3.append(0)
            elif (c == 'm') or (c == 'M'):
                xi_2.append(np.exp(2*np.pi*1j*2/7))
                xi_0.append(0)
                xi_1.append(0)
                xi_3.append(0)
            elif (c == 'n') or (c == 'N'):
                xi_2.append(np.exp(2*np.pi*1j*3/7))
                xi_0.append(0)
                xi_1.append(0)
                xi_3.append(0)
            elif (c == 'p') or (c == 'P'):
                xi_2.append(np.exp(2*np.pi*1j*4/7))
                xi_0.append(0)
                xi_1.append(0)
                xi_3.append(0)
            elif (c == 'q') or (c == 'Q'):
                xi_2.append(np.exp(2*np.pi*1j*5/7))
                xi_0.append(0)
                xi_1.append(0)
                xi_3.append(0)
            elif (c == 'r') or (c == 'R'):
                xi_2.append(np.exp(2*np.pi*1j*6/7))
                xi_0.append(0)
                xi_1.append(0)
                xi_3.append(0)

        elif (c in ['s', 't','v', 'w', 'x', 'y', 'z']) or (c in ['S', 'T',' V', 'W', 'X', 'Y', 'Z']): 
            l0.append(0)
            l1.append(0)
            l2.append(0)
            l3.append(1)
            # XI ENCODING
            if (c == 's') or (c == 'S'):
                xi_3.append(np.exp(2*np.pi*1j*0/7))
                xi_0.append(0)
                xi_1.append(0)
                xi_2.append(0)
            elif (c == 't') or (c == 'T'):
                xi_3.append(np.exp(2*np.pi*1j*1/7))
                xi_0.append(0)
                xi_1.append(0)
                xi_2.append(0)
            elif (c == 'v') or (c == 'V'):
                xi_3.append(np.exp(2*np.pi*1j*2/7))
                xi_0.append(0)
                xi_1.append(0)
                xi_2.append(0)
            elif (c == 'w') or (c == 'W'):
                xi_2.append(np.exp(2*np.pi*1j*3/7))
                xi_0.append(0)
                xi_1.append(0)
                xi_2.append(0)
            elif (c == 'x') or (c == 'X'):
                xi_3.append(np.exp(2*np.pi*1j*4/7))
                xi_0.append(0)
                xi_1.append(0)
                xi_2.append(0)
            elif (c == 'y') or (c == 'Y'):
                xi_3.append(np.exp(2*np.pi*1j*5/7))
                xi_0.append(0)
                xi_1.append(0)
                xi_2.append(0)
            elif (c == 'z') or (c == 'Z'):
                xi_3.append(np.exp(2*np.pi*1j*6/7))
                xi_0.append(0)
                xi_1.append(0)
                xi_2.append(0)
        else:
            print('Input word contain character our of English alphabet, the word is: {}'.format(c))
    


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
    pe = np.concatenate([l0,l1,l2,l3], axis = 0)
    fe = np.concatenate([xi_0,xi_1,xi_2,xi_3], axis = 0)
    

    return pe, fe
