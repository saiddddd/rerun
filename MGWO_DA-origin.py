import numpy as np
import pandas as pd
from numpy.random import rand
from Function_update import *
import math
import random

random.seed(42)
sed = random.random()

def init_position(lb, ub, N, dim):
    lowerbound = np.ones(dim) * lb # Lower limit for variables
    upperbound = np.ones(dim) * ub # Upper limit for variables

    # INITIALIZATION
    X = np.random.uniform(lowerbound, upperbound, (N, dim))  # Initial population

    return X

def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            result = 1 / (1 + math.exp(-10 * (X[i, d] - 0.5)))
            if result > sed:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0
        if np.all(Xbin[i, :] == 0):
            for d in range(dim):
                result = 1 / (1 + math.exp(-10 * (X[i, d] - 0.5)))
                if result > sed:
                    Xbin[i, d] = 1
                else:
                    Xbin[i, d] = 0
        if np.all(Xbin[i, :] == 1):
            for d in range(dim):
                result = 1 / (1 + math.exp(-10 * (X[i, d] - 0.5)))
                if result > sed:
                    Xbin[i, d] = 1
                else:
                    Xbin[i, d] = 0
    return Xbin

def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub

    return x

def binary_conversion2(X, thres, N, dim, sed):
    Xbin = np.zeros([N, dim], dtype='int')
    if len(X.shape) == 1:
        Xbin = (X > thres).astype(int)
        if np.sum(Xbin) == 0:
            Xbin[np.random.randint(0, dim)] = 1
    else:
        for i in range(N):
            Xbin[i, :] = (X[i, :] > thres).astype(int)
            if np.sum(Xbin[i, :]) == 0:
                Xbin[i, np.random.randint(0, dim)] = 1
    return Xbin

def jfs(xtrain, ytrain, opts):
    ub = 1
    lb = 0
    thres = 0.5
    mark = 0
    w = [1, 1]
    no_promoted = 0
    a_list = []
    alpha = 2
    F = 0.5
    max_index = 0

    N = opts['N']
    max_iter = opts['T']

    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    X = init_position(lb, ub, N, dim)

    Xbin = binary_conversion(X, thres, N, dim)

    fit = np.zeros([N, 1], dtype='float')
    
    for i in range(N):
        fit[i, 0], num, error = Fun(xtrain, ytrain, Xbin[i, :], opts)
    
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    curve[0, t] = fit[i,0].min()
    t += 1

    curve[0, t] = fit[i,0].min()
    t += 1

    Fbest = curve[0, t - 1]  # Initial value of Fbest
    PBest = X[0, :]  # Initial value of PBest
    Pweak = X[0, :]  # Inisialisasi Pweak

    while t < max_iter:
        flag = 0
        num_list = []
        fit_list = []
        
        best, location, weak_location = fit[:, 0].min(), fit[:, 0].argmin(), fit[:, 0].argmax()
        if t == 0:
            PBest = X[location, :]  # Optimal location
            Fbest = best  # The optimization objective function
            Pweak = X[location, :] 
        elif best < Fbest:
            Fbest = best
            PBest = X[location, :]
        elif best > Fbest:
            weak = best
            Pweak = X[weak_location, :]
        
        # PHASE 1: Foraging Behaviour
        for i in range(N):
            I = np.round(1 + np.random.rand())
            X_newP1 = X[i, :] + np.random.rand(dim) * (PBest - I * X[i, :]) + np.random.rand(dim)*(PBest - I*Pweak) 
          
            X_newP1 = np.maximum(X_newP1, lb)
            X_newP1 = np.minimum(X_newP1, ub)
            X[i, :] = X_newP1
            
            B = X[i, :].copy()
            B[B > 1] = 1
            B[B < 0] = 0
            X[i, :] = B

        for i in range(N):
            fit[i, 0], num, error = Fun(xtrain, ytrain, X[i, :], opts)
            num_list.append(num)
            fit_list.append(fit[i, 0])
            list_a = fit.tolist()
            max_index = list_a.index(max(list_a))
            if fit[i, 0] < Fbest:
                flag = 1
                PBest = X[i, :]
                Fbest = fit[i, 0]
                mark = i
        # End Phase 1: Foraging Behaviour
        
        # PHASE 2: Collaboration
        for i in range(N):            
            I = np.round(1 + np.random.rand())
            X_newP2 = (X[i, :] + ((lb - ub)*PBest*np.random.standard_cauchy()) * (PBest - I*X[i, :])
                      )/(N)
            X_newP2 = 0.007241*X_newP2*np.random.rand()*(PBest - Pweak) + X_newP2*np.random.standard_cauchy()

            X_newP2 = np.maximum(X_newP2, lb)
            X_newP2 = np.minimum(X_newP2, ub)

            X[i, :] = X_newP2
            
            B = X[i, :].copy()
            B[B > 1] = 1
            B[B < 0] = 0
            X[i, :] = B

        for i in range(N):
            fit[i, 0], num, error = Fun(xtrain, ytrain, X[i, :], opts)
            num_list.append(num)
            fit_list.append(fit[i, 0])
            list_a = fit.tolist()
            max_index = list_a.index(max(list_a))
            if fit[i, 0] < Fbest:
                flag = 1
                PBest = X[i, :]
                Fbest = fit[i, 0]
                mark = i
        #End Phase 2: Collaboration       

        # PHASE 3: Random moving
        for i in range(N):     
            I = np.round(1 + np.random.rand())
            X_newP3 = X[i, :] + \
            np.random.random()*(PBest - I*abs(X[i, :]))/N - (Pweak - I*abs(X[i,:]))/N
            

            X_newP3 = np.maximum(X_newP3, lb)
            X_newP3 = np.minimum(X_newP3, ub)
        
            
            X[i, :] = X_newP3
            
            B = X[i, :].copy()
            B[B > 1] = 1
            B[B < 0] = 0
            X[i, :] = B

        for i in range(N):
            fit[i, 0], num, error = Fun(xtrain, ytrain, X[i, :], opts)
        
            num_list.append(num)
            fit_list.append(fit[i, 0])
            list_a = fit.tolist()
            max_index = list_a.index(max(list_a))
            if fit[i, 0] < Fbest:
                flag = 1
                PBest = X[i, :]
                Fbest = fit[i, 0]
                mark = i
        # End Phase 3: Random moving          
       
        # PHASE 4: Defense strategies against predators
        for i in range(N):
            Ps = np.random.rand()
            k = np.random.randint(N)
            A = X[k, :]  # Attacked

            r1 = np.random.random()
            r2 = np.random.random()
            for i in range(N):
                if Ps <= 0.5:

                    R = 0.7 #0.9
                    I = np.round(1 + np.random.rand())
                    X_newP4 = X[i, :] + r2 * (PBest - I * X[i, :]) + ((2 * np.random.rand(dim) - 1)) * (2 - ((t)/ (max_iter))) 

                    X_newP4 = np.maximum(X_newP4, lb)
                    X_newP4 = np.minimum(X_newP4, ub)
                else:
                    X_newP4 = X[i, :] + np.random.rand(dim) * (A - I * X[i, :]) + np.random.rand(dim)*(PBest - I*Pweak)
                    X_newP4 = X_newP4 + X_newP4*np.random.rand(dim) - X_newP4*np.random.standard_cauchy()

                    X_newP4 = np.maximum(X_newP4, lb)
                    X_newP4 = np.minimum(X_newP4, ub)   
            
            X[i, :] = X_newP4
            
            B = X[i, :].copy()
            B[B > 1] = 1
            B[B < 0] = 0
            X[i, :] = B

        for i in range(N):
            fit[i, 0], num, error = Fun(xtrain, ytrain, X[i, :], opts)
            num_list.append(num)
            fit_list.append(fit[i, 0])
            list_a = fit.tolist()
            max_index = list_a.index(max(list_a))
            if fit[i, 0] < Fbest:
                flag = 1
                PBest = X[i, :]
                Fbest = fit[i, 0]
                mark = i
        # PHASE 4: Defense strategies against predators      
    
        
        B = np.zeros([3, dim], dtype='float32')
        BN = np.zeros([3, dim], dtype='float32')
        B[0, :] = PBest
        B[1, :] = PBest
        B[2, :] = PBest
        BN[0, :] = 1 - PBest
        BN[1, :] = 1 - PBest
        BN[2, :] = 1 - PBest

        for i in range(3):
            RN = np.random.permutation(3)
            for j in range(3):
                if RN[j] == i:
                    RN = np.delete(RN, j)
                    break
            r1 = RN[0]
            r2 = RN[1]
            B[i, :] = PBest + F * (B[r1, :] - B[r2, :]) + F * (B[i, :] - X[max_index, :])
            for d in range(dim):
                B[i, d] = B[i, d] + random.uniform(-(1 - t / max_iter), 0)

        B[B > 1] = 1
        B[B < 0] = 0
        Bbin = binary_conversion2(B, thres, 3, dim, sed)

        for i in range(3):
            fit[i, 0], num, error = Fun(xtrain, ytrain, Bbin[i, :], opts)
            if fit[i, 0] < Fbest:
                PBest = B[i, :]
                Fbest = fit[i, 0]

        curve[0, t] = Fbest.copy()
        t = t + 1

    Gbin = binary_conversion2(PBest, thres, 1, dim, sed)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)

    sgho_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}

    return sgho_data
