# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 15:17:02 2022

Play with covariance matrices, and how to change the Participation Ratio by modifying some elements

@author: Olivia Gozel
"""


import numpy as np
import matplotlib.pyplot as plt



#%% Compute PR from the covariance matrix

def computePR(C):
    
    # Perform Singluar value decomposition
    u,s,__ = np.linalg.svd(C)
    
    # Compute Participation Ratio
    this_PR = np.power(np.sum(s),2)/np.sum(np.power(s,2))
    
    #print('Participation ratio = '+str(round(this_PR,1)))
    
    return this_PR
    


#%% Compute PR for a covariance matrix (C) and modifications of C several times

# NB: If we increase (or decrease) all elements of C with the same gain, it does not change the PR

N = 100 # number of variables
nDraws = 100

pr_C = []
pr_C_incrVar = []
pr_C_decrVar = []
pr_C_incrCov = []
pr_C_decrCov = []

for d in range(nDraws):

    tmp = 2*np.random.random(size=(N,N)) - np.ones(shape=(N,N)) # random values between - 1 and 1
    
    # Create a covariance matrix
    C = (tmp + tmp.T)/2 # symmetric
    np.fill_diagonal(C,np.random.random(size=(N,N))) # diagonal is positive
    
    # Compute PR
    pr_C.append( computePR(C) )
    
    # Increase the variance (diagonal elements) only ---------------------------------------- decrease of PR
    C_incrVar = C + np.diag(5*np.random.random(size=(N,)))
    pr_C_incrVar.append( computePR(C_incrVar) )
    
    # Decrease the variance (diagonal elements) only (it should still stay above 0) --------- slight increase in PR
    C_decrVar = C - np.diag(0.8*np.diag(C))
    pr_C_decrVar.append( computePR(C_decrVar) )
    
    # Increase the covariance (non-diagonal elements) only  --------------------------------- slight increase of PR
    C_incrCov = 5*C
    np.fill_diagonal(C_incrCov,np.diag(C))
    pr_C_incrCov.append( computePR(C_incrCov) )
    
    # Decrease the covariance (non-diagonal elements) only  --------------------------------- decrease of PR
    tmp = 0.8*C
    np.fill_diagonal(tmp,np.zeros(shape=(N,)))
    C_decrCov = C - tmp
    pr_C_decrCov.append( computePR(C_decrCov) )



#%% Plotting

plt.figure()
meanPR = [np.mean(pr_C), np.mean(pr_C_incrVar), np.mean(pr_C_decrVar), np.mean(pr_C_incrCov), np.mean(pr_C_decrCov)]
semPR = [np.std(pr_C)/np.sqrt(nDraws), np.std(pr_C_incrVar)/np.sqrt(nDraws), np.std(pr_C_decrVar)/np.sqrt(nDraws), np.std(pr_C_incrCov)/np.sqrt(nDraws), np.std(pr_C_decrCov)/np.sqrt(nDraws)]
plt.errorbar(np.arange(5),meanPR,yerr=semPR,linestyle='')
plt.xticks(np.arange(5),labels=['C','Incr Var', 'Decr Var', 'Incr Cov', 'Decr Cov'])
plt.ylabel('PR')

