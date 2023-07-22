import torch
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
import mindquantum as mq
from mindquantum.core.gates import H, RY, RX, RZ

def OneHotEncoding(x):
    X = []
    for i in range(len(x)):
        if x[i] == 'A':
            X.append(torch.tensor([1,0,0,0]).unsqueeze(0))
        elif x[i] == 'T':
            X.append(torch.tensor([0,1,0,0]).unsqueeze(0))
        elif x[i] == 'G':
            X.append(torch.tensor([0,0,1,0]).unsqueeze(0))
        elif x[i] == 'C':
            X.append(torch.tensor([0,0,0,1]).unsqueeze(0))
        else:
            print('Invalid RNA sequences')
    X = torch.cat(X, dim = 0)
    return X

def LazyEncoding(x):
    X = []
    for i in range(len(x)):
        if x[i] == 'A':
            X.append(torch.tensor([-np.pi/2, -np.pi/2]).unsqueeze(0))
        elif x[i] == 'T':
            X.append(torch.tensor([-np.pi/2, np.pi/2]).unsqueeze(0))
        elif x[i] == 'G':
            X.append(torch.tensor([np.pi/2, -np.pi/2]).unsqueeze(0))
        elif x[i] == 'C':
            X.append(torch.tensor([np.pi/2, np.pi/2]).unsqueeze(0))
        else:
            print('Invalid RNA sequences')
    X = torch.cat(X, dim = 0)
    return X


def root(k):
    x = np.exp(2*1j*k*np.pi/3)
    return x

def DynamicalCyclicEncoding(x):
    X = []
    _X = []
    for i in range(len(x)-3):
        _bm = x[i]
        _b = [x[i+j] for j in range(1,4)]
        if _bm == 'A':
            # C = -1, T = 0, G = 1
            _bm = root(3/2)**1
            for i in range(len(_b)):
                if _b[i] == 'C':
                    _b[i] = root(-1)
                elif _b[i] == 'T':
                    _b[i] = root(0)
                elif _b[i] == 'G':
                    _b[i] = root(1)
                else:
                    _b[i] =root(3/2)
            #X.append(torch.tensor(_bm))
            #_X.append(torch.tensor(np.array(_b)).reshape(-1,1))
        elif _bm == 'T':
            # C = 1, A = 0, G = -1
            _bm = root(3/2)**0
            for i in range(len(_b)):
                if _b[i] == 'C':
                    _b[i] = root(1)
                elif _b[i] == 'A':
                    _b[i] = root(0)
                elif _b[i] == 'G':
                    _b[i] = root(-1)
                else:
                    _b[i] =root(3/2)
            #X.append(torch.tensor(_bm))
            #_X.append(torch.tensor(np.array(_b)).reshape(-1,1))
        elif _bm == 'C':
            # A = -1, G = 0, T = 1
            _bm = root(3/2)**0
            for i in range(len(_b)):
                if _b[i] == 'A':
                    _b[i] = root(-1)
                elif _b[i] == 'G':
                    _b[i] = root(0)
                elif _b[i] == 'T':
                    _b[i] = root(1)
                else:
                    _b[i] =root(3/2)    
            #X.append(torch.tensor(_bm))
            #_X.append(torch.tensor(np.array(_b)).reshape(-1,1))
        elif _bm == 'G':
            # A = 1, C = 0, T = -1
            _bm = root(3/2)**1
            for i in range(len(_b)):
                if _b[i] == 'A':
                    _b[i] = root(1)
                elif _b[i] == 'G':
                    _b[i] = root(0)
                elif _b[i] == 'T':
                    _b[i] = root(-1)
                else:
                    _b[i] =root(3/2)
        X.append(torch.tensor(_bm))
        _X.append(torch.tensor(np.array(_b)).unsqueeze(0))
    r = torch.tensor(X)
    _X = torch.cat(_X, dim = 0)
    R = torch.diag(r)
    n = torch.matmul(R,_X)
    N = torch.cat([r.real.unsqueeze(1), n.angle().float()], dim = 1)
    return N


def QuantumEmbeddingR1(x):
    _X = []
    for i in range(len(x)):
        theta_0 = x[i][0]
        theta_1 = x[i][1]
        rx = RX('alpha')
        ry = RY('beta')
        rz = RZ('gamma')
        RotX0 = rx.matrix({'alpha': theta_0.numpy()})
        RotX1 = rx.matrix({'alpha': theta_1.numpy()})
        RotY0 = ry.matrix({'beta': theta_0.numpy()})
        RotY1 = ry.matrix({'beta': theta_1.numpy()})
        RotZ0 = rz.matrix({'gamma': theta_0.numpy()})
        RotZ1 = rz.matrix({'gamma': theta_1.numpy()})
        SRX = np.concatenate([RotX0, RotX1], axis = 1)
        SRY = np.concatenate([RotY0, RotY1], axis = 1)
        SRZ = np.concatenate([RotZ0, RotZ1], axis = 1)
        SR = np.concatenate([SRX, SRY, SRZ], axis = 0)
        _X.append(SR)
    _X = np.array(_X)
    _X = np.abs(_X)
    return _X


def QuantumEmbeddingR2(x):
    _X = []
    for i in range(len(x)):
        zeta_0=x[i][1]
        zeta_1=x[i][2]
        zeta_2=x[i][3]
        rx = RX('theta_0')
        ry = RY('theta_1')
        rz = RZ('theta_2')
        RotX0 = rx.matrix({'theta_0': zeta_0.numpy()})
        RotX1 = rx.matrix({'theta_0': zeta_1.numpy()})
        RotX2 = rx.matrix({'theta_0': zeta_2.numpy()})
        RotY0 = ry.matrix({'theta_1': zeta_0.numpy()})
        RotY1 = ry.matrix({'theta_1': zeta_1.numpy()})
        RotY2 = ry.matrix({'theta_1': zeta_2.numpy()})
        RotZ0 = rz.matrix({'theta_2': zeta_0.numpy()})
        RotZ1 = rz.matrix({'theta_2': zeta_1.numpy()})
        RotZ2 = rz.matrix({'theta_2': zeta_2.numpy()})
        SRX = np.concatenate([RotX0, RotX1, RotX2], axis = 1)
        SRY = np.concatenate([RotY0, RotY1, RotY2], axis = 1)
        SRZ = np.concatenate([RotZ0, RotZ1, RotZ2], axis = 1)
        SR = np.concatenate([SRX, SRY, SRZ], axis = 0)
        _X.append(SR)
    _X = np.array(_X)
    return _X