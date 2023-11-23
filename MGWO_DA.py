import numpy as np
import pandas as pd
from numpy.random import rand
from Function_update import *
import math
import random

random.seed(233)
sed = random.random()


def init_position(lb, ub, N, dim):
    Xh = np.zeros([N, int(dim * 0.05)], dtype='float')
    Xm = np.zeros([N, int(dim * 0.35)], dtype='float')
    Xl = np.zeros([N, (dim - (int(dim * 0.05) + int(dim * 0.35)))], dtype='float')
    for i in range(N):
        random.seed()
        l = int(random.uniform(0, int(dim * 0.2)))
        mid_number = int(l * random.uniform(0.8, 1))
        low_number = l - mid_number

        Xh[i, :] = random.uniform(0.5, 1)

        sample_list_m_1 = [i for i in range(int(dim * 0.35))]
        if len(sample_list_m_1) < mid_number:
            random.seed()
            Xm[i, :] = random.uniform(0.5, 1)
        else:
            sample_list_m = random.sample(sample_list_m_1, mid_number)
            random.seed()
            Xm[i, sample_list_m] = random.uniform(0.5, 1)

        sample_list_l = [i for i in range((dim - (int(dim * 0.05) + int(dim * 0.35))))]
        if len(sample_list_l) < low_number:
            random.seed()
            Xl[i, :] = random.uniform(0.5, 1)
        else:
            sample_list_l = random.sample(sample_list_l, low_number)
            random.seed()
            Xl[i, sample_list_l] = random.uniform(0.5, 1)
    X = np.hstack((Xh, Xl, Xm))

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


def binary_conversion2(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        Xbin[i, :] = X[i, :].copy()
        Xbin[i, :] = 1 * (1 / (1 + np.exp(-10 * (X[i, :] - 0.5))) >= sed)
        if np.all(Xbin[i, :] == 0):
            Xbin[i, :] = 1 * (1 / (1 + np.exp(-10 * (X[i, :] - 0.5))) >= sed)
        if np.all(Xbin[i, :] == 1):
            Xbin[i, :] = 1 * (1 / (1 + np.exp(-10 * (X[i, :] - 0.5))) >= sed)
    return Xbin 


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub

    return x


def quasi_opposite_learning(position, lowerbound, upperbound):
    new_position = []
    for i in  position:
        new_position.append(np.random.uniform((lowerbound+upperbound)/2, lowerbound + upperbound - i))
    return new_position    
   

def opposite_learning(position, lowerbound, upperbound):
    opposite_position = upperbound - np.abs(position*np.random.standard_cauchy() - lowerbound)
    return opposite_position


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
    fit1 = np.zeros([N, 1], dtype='float')
    fit2 = np.zeros([N, 1], dtype='float')
    fit3 = np.zeros([N, 1], dtype='float')
    Xalpha = np.zeros([1, dim], dtype='float')
    Xbeta = np.zeros([1, dim], dtype='float')
    Xdelta = np.zeros([1, dim], dtype='float')
    Falpha = float('inf')
    Fbeta = float('inf')
    Fdelta = float('inf')
    for i in range(N):
        fit[i, 0], num, error = Fun(xtrain, ytrain, Xbin[i, :], opts)
        if fit[i, 0] < Falpha:
            Xalpha[0, :] = X[i, :]
            Falpha = fit[i, 0]
            mark = i

        if Fbeta > fit[i, 0] > Falpha:
            Xbeta[0, :] = X[i, :]
            Fbeta = fit[i, 0]

        if Fdelta > fit[i, 0] > Fbeta and fit[i, 0] > Falpha:
            Xdelta[0, :] = X[i, :]
            Fdelta = fit[i, 0]

    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    curve[0, t] = Falpha.copy()
    t += 1

    # Define mutation parameters for non-uniform mutation
    initial_mutation_rate = 0.2  # Initial mutation rate
    final_mutation_rate = 0.01  # Final mutation rate
    #mutation_decay = 0.9  # Rate of mutation decay (adjust as needed)

    # Define OLB parameters
    olb_probability = 0.2  # Probability of applying OLB
    
    
    while t < max_iter:
        flag = 0
        num_list = []
        fit_list = []
        a_max = 2
        a_min = 0
        a = np.sin(2 * math.pi * t / max_iter)
        
        # Calculate the mutation rate for non-uniform mutation
        mutation_rate = initial_mutation_rate * ((final_mutation_rate / initial_mutation_rate) ** (t / max_iter))


        for i in range(N):
            r1 = np.random.uniform(size=dim)
            r2 = np.random.uniform(size=dim)

            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D1 = np.abs(C1 * Xalpha - X[i, :])
            
            X1 = Xalpha - A1 * D1
            
            r3 = np.random.uniform(size=dim)
            r4 = np.random.uniform(size=dim)
            A2 = 2 * a * r3 - a
            C2 = 2 * r4
            D2 = np.abs(C2 * Xbeta - X[i, :])
            
            X2 = Xbeta - A2 * D2
            
            r5 = np.random.uniform(size=dim)
            r6 = np.random.uniform(size=dim)
            A3 = 2 * a * r5 - a
            C3 = 2 * r6
            D3 = np.abs(C3 * Xdelta - X[i, :])
            
            X3 = Xdelta - A3 * D3
            
            del r1, r2, r3, r4, r5, r6
            
            
            if np.all(np.random.uniform(lb, ub) < np.std(np.abs(X1 - X2))) and \
                           np.all(np.random.uniform(lb, ub) < np.std(np.abs(X1 - X3))):
                            # Calculate the distances between X1 and X2, and between X1 and X3
                            dist_X1_X2 = np.abs(X1 - X2)
                            dist_X1_X3 = np.abs(X1 - X3)

                            # If the distances are greater than a certain threshold, move X2 and X3 closer to X1
                            threshold = 0.5  # Set the threshold to a suitable value
                            if np.all(dist_X1_X2 > threshold) and np.all(dist_X1_X3 > threshold):
                                random_factor = np.random.uniform(-1, 1) 

                                X2 = X2 - (abs(X2 - X1) / (ub - lb)) * np.random.uniform(lb, ub)*random_factor
                                X3 = X3 - (abs(X3 - X1) / (ub - lb)) * np.random.uniform(lb, ub)*random_factor            
            
            X[i, :] = (X1*0.7 +  0.2 * X2 + 0.1 * X3)
            
            # Apply Non-Uniform Mutation
            if np.random.rand() < mutation_rate:
                #wolves[i] = wolves[i] + np.random.normal(0, 1)  
                random_indexed = random.randint(0, len(X[i, :]) - 1)
                
                cross_thrs = random.choice([1, -1])
                
                snum = (1 - np.random.uniform(0, 1) ** ((1 - (t / max_iter))) ** (1/2))

                if cross_thrs == 1:
                    X[i, random_indexed] -= (ub[0][0] - X[i, random_indexed]) * snum
                    
                elif cross_thrs == -1:
                    X[i, random_indexed] -= (X[i, random_indexed] - lb[0][0]) * snum
                #wolves[i] = wolves[i] + np.random.standard_cauchy()
                
                
            # Limit the wolf's position within the search bounds if necessary
            X[i, :] = np.maximum(X[i, :], lb)
            X[i, :] = np.minimum(X[i, :], ub)
            
            #Apply Opposite Learning Based (OLB)
            if np.random.rand() < olb_probability:
                X[i, :] = opposite_learning(X[i, :], lb[0][0], ub[0][0])
            
            X[i, :] = np.maximum(X[i, :], lb)
            X[i, :] = np.minimum(X[i, :], ub)
            
            
            if np.random.rand() < 0.2:
                X[i, :] = quasi_opposite_learning(X[i, :], lb[0][0], ub[0][0])
            
            X[i, :] = np.maximum(X[i, :], lb)
            X[i, :] = np.minimum(X[i, :], ub)
            
            B = X[i, :].copy()
            B[B > 1] = 1
            B[B < 0] = 0
            X[i, :] = B

        Xbin = binary_conversion2(X, thres, N, dim)
        alpha_advance = 0
        beta_advance = 0
        delta_advance = 0

        for i in range(N):
            fit[i, 0], num, error = Fun(xtrain, ytrain, Xbin[i, :], opts)
            num_list.append(num)
            fit_list.append(fit[i, 0])
            list_a = fit.tolist()
            max_index = list_a.index(max(list_a))
            if fit[i, 0] < Falpha:
                flag = 1
                alpha_advance = Falpha - fit[i, 0]
                Xalpha[0, :] = X[i, :]
                Falpha = fit[i, 0]
                mark = i

            if Fbeta > fit[i, 0] > Falpha:
                beta_advance = Fbeta - fit[i, 0]
                Xbeta[0, :] = X[i, :]
                Fbeta = fit[i, 0]

            if Fdelta > fit[i, 0] > Fbeta and fit[i, 0] > Falpha:
                delta_advance = Fdelta - fit[i, 0]
                Xdelta[0, :] = X[i, :]
                Fdelta = fit[i, 0]

        if beta_advance or delta_advance != 0:
            w[0] = beta_advance / (beta_advance + delta_advance)
            w[1] = delta_advance / (beta_advance + delta_advance)
        else:
            w[0] = 1
            w[1] = 1
        if alpha_advance == 0:
            no_promoted = no_promoted + 1
        else:
            no_promoted = 0

        B = np.zeros([3, dim], dtype='float32')
        BN = np.zeros([3, dim], dtype='float32')
        B[0, :] = Xalpha[0, :]
        B[1, :] = Xbeta[0, :]
        B[2, :] = Xdelta[0, :]
        BN[0, :] = 1 - Xalpha[0, :]
        BN[1, :] = 1 - Xbeta[0, :]
        BN[2, :] = 1 - Xdelta[0, :]

        for i in range(3):
            RN = np.random.permutation(3)
            for j in range(3):
                if RN[j] == i:
                    RN = np.delete(RN, j)
                    break
            r1 = RN[0]
            r2 = RN[1]
            B[i, :] = Xalpha[0, :] + F * (B[r1, :] - B[r2, :]) + F * (B[i, :] - X[max_index, :])
            for d in range(dim):
                B[i, d] = B[i, d] + random.uniform(-(1 - t / max_iter), 0)
                B[1, d] = B[1, d] + F * (BN[0, d] - BN[2, d]) + F * (Xalpha[0, d] - X[max_index, d])
                B[2, d] = B[2, d] + F * (BN[0, d] - BN[1, d]) + F * (Xalpha[0, d] - X[max_index, d])

        B[B > 1] = 1
        B[B < 0] = 0
        Bbin = binary_conversion2(B, thres, 3, dim)

        for i in range(3):
            fit[i, 0], num, error = Fun(xtrain, ytrain, Bbin[i, :], opts)
            if fit[i, 0] < Falpha:
                Xalpha[0, :] = B[i, :]
                Falpha = fit[i, 0]
                mark = i

            if Fbeta > fit[i, 0] > Falpha:
                beta_advance = Fbeta - fit[i, 0]
                Xbeta[0, :] = B[i, :]
                Fbeta = fit[i, 0]

            if Fdelta > fit[i, 0] > Fbeta and fit[i, 0] > Falpha:
                delta_advance = Fdelta - fit[i, 0]
                Xdelta[0, :] = B[i, :]
                Fdelta = fit[i, 0]

        if beta_advance or delta_advance != 0:
            w[0] = beta_advance / (beta_advance + delta_advance)
            w[1] = delta_advance / (beta_advance + delta_advance)
        else:
            w[0] = 1
            w[1] = 1

        curve[0, t] = Falpha.copy()
        t = t + 1

    Gbin = binary_conversion2(Xalpha, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)

    gwo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}

    return gwo_data


