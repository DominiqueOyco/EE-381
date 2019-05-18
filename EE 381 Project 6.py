#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:16:55 2019

@author: rain
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import matplotlib.lines as mlines     

#1 - three state markov chain - single simulated, simulated, and calculated
def threeStateMarkovChain(N,n):
    S = []                      #container for the states
    X = np.chararray((n,N))     #store the used to store the chars R, N, and S
    X[:] = 0                    #initializing the contents of the character array to 0
    M = np.zeros((n,3))         #will be used to plot the probability of each chars
    
    stateProb = {'p11' : 1/2, 'p12' : 1/4, 'p13' : 1/4,
                 'p21' : 1/4, 'p22' : 1/8, 'p23' : 5/8,
                 'p31' : 1/3, 'p32' : 2/3, 'p33' : 0}
    stateDist = {'d1' : 1/4, 'd2' : 0, 'd3' : 3/4}
    
    for i in range(0,n):
        S.append(0)
    
    for j in range(0,N):
        randNum = random.random()   #return a value from 0 to 1 not including 1
        
        if(randNum <= stateDist['d1']):
            currentState = 'R'
        elif(randNum > stateDist['d1'] and randNum <= stateDist['d1'] + stateDist['d2']):
            currentState = 'N'
        elif(randNum > stateDist['d1'] + stateDist['d2']):
            currentState = 'S'
        S[0] = currentState
        
        for k in range(0,n - 1):
            s = S[k]
            r = random.random()     #return a value from 0 to 1 not including 1    
            
            if s == 'R':
                if(r <= stateProb['p11']):
                    S[k+1]='R'
                elif(r > stateProb['p11'] and r <= stateProb['p11'] + stateProb['p12']):
                    S[k+1]='N'
                elif(r > stateProb['p11'] + stateProb['p12']):
                    S[k+1]='S'
                    
            elif s == 'N':
                if(r <= stateProb['p21']):
                    S[k+1]='R'
                elif(r > stateProb['p21'] and r <= stateProb['p21'] + stateProb['p22']):
                    S[k+1]='N'
                elif(r > stateProb['p21'] + stateProb['p22']):
                    S[k+1]='S'
                    
            elif s == 'S':
                if(r <= stateProb['p31']):
                    S[k+1]='R'
                elif(r >= stateProb['p31'] and r <= stateProb['p31'] + stateProb['p32']):
                    S[k+1]='N'
                elif(r > stateProb['p31'] + stateProb['p32']):
                    S[k+1]='S'
        X[:,j]=S
        
    for l in range(0,n):
        x = X[l,:]
        mr = 0
        mn = 0
        ms = 0
        
        for m in range(N):
            if str(x[m],'utf-8') == 'R':
                mr += 1
        for o in range(N):
            if str(x[o],'utf-8') == 'N':
                mn += 1
        for p in range(N):
            if str(x[p],'utf-8') == 'S':
                ms += 1
                
        M[l,:] = [mr/N,mn/N,ms/N]
        
    # Figure 1 - graph for single simulated Markov Chain 
    plt.figure(1)
#    plt.yticks([1, 2, 3])
    plt.scatter(np.arange(len(S)), S, color='r', edgecolors='b')
    plt.xticks(np.arange(len(S)), np.arange(1, len(S) + 1))
    plt.plot(S, 'b:')
    plt.ylabel('States')
    plt.xlabel('Step Number')
    plt.title('Simulation Run of a Three State Markov Chain')
    plt.show()

    # Figure 2 - graph for the simulated Markov Chain
    plt.figure(2)
    nv = np.linspace(0, n, num = n) 
    plt.plot(nv, M[:,0], 'c--', marker='o')
    plt.plot(nv, M[:,1], 'g--', marker='o')
    plt.plot(nv, M[:,2], 'm--', marker='o')
    
    plt.title('Simulated three-state Markov Chain')
    plt.xlabel('Step Number')
    plt.ylabel('Probability')
    plt.legend(('State R','State N','State S'))
    plt.show()
                                          
    # Figure 3 -  graph for the calculated Markov Chain
    P = np.matrix([[stateProb['p11'],stateProb['p12'],stateProb['p13']],
               [stateProb['p21'],stateProb['p22'],stateProb['p23']],
               [stateProb['p31'],stateProb['p32'],stateProb['p33']]])
    
    y0 = np.matrix([1/4, 0, 3/4]) #initial distribution vector
    
    Y = np.zeros((n,3))
    Y[0,:] = y0
    
    for k in range(0,n-1):
        Y[k+1,:] = np.matmul(Y[k,:],P) #multiply the y0 matrix by the P matrix 
                                       # and return the matrix product     
    plt.figure(3)
    plt.plot(nv, Y[:,0], 'c--', marker='o')
    plt.plot(nv, Y[:,1], 'g--', marker='o')
    plt.plot(nv, Y[:,2], 'm--', marker='o')
    
    plt.title('Calculated three-state Markov Chain')
    plt.xlabel('Step Number')
    plt.ylabel('Probability')
    plt.legend(('State R','State N','State S'))
    plt.show()
    
threeStateMarkovChain(10000,15)

#2
def pageRank(N,n):    
    #state transition matrix
    P = np.matrix([[0, 1, 0, 0, 0], [1/2, 0, 1/2, 0, 0],[1/3, 1/3, 0, 0, 1/3],
                   [1, 0, 0, 0, 0],[0, 1/3, 1/3, 1/3, 0]])
    
    #initial distribution vector
    V = np.matrix([[0.2, 0.2, 0.2, 0.2, 0.2],[0, 0, 0, 0, 1]])
    
    for i in range(0,2):
        currentState = V[i,:]
        Y = np.zeros((n,5))
        Y[0,:] = currentState

        for j in range(0,n-1):
            Y[j+1,:] = np.matmul(Y[j,:],P) #multiply the current row in the matrix 
                                           #by the each column in the state transition matrix
                                           #works with stacks of matrices but not scalar 
                                           #multiplication
            
        nv = np.linspace(0, n, num=n)
        plt.figure()
        plt.plot(nv, Y, marker = 'o')
        plt.title(('Page Rank Probabilities: V = ', np.str(currentState)))
        plt.xlabel('Step Number')
        plt.ylabel('Page Visit Probability')
        plt.legend(('A','B','C','D','E'))
        plt.show
        
pageRank(10000,20)

#3
def nSidedDie(p):
    n = len(p)  
    cs = np.cumsum(p)
    cp = np.append(0, cs)

    r = np.random.rand()
    for k in range(0, n):

        if r > cp[k] and r <= cp[k + 1]:
            d = k

    return d

def fiveStateAbsorbingMarkovChain(n):
    S = np.zeros(n)
    randNum = nSidedDie([0, 0.33, 0.33, 0.33, 0])
    S[0] = randNum

    for i in range(n-1):
        if randNum == 0:
            randNum = nSidedDie([1, 0, 0, 0, 0])
            S[i+1] = randNum

        elif randNum == 1:
            randNum = nSidedDie([0.3, 0, 0.7, 0, 0])
            S[i+1] = randNum

        elif randNum == 2:
            randNum = nSidedDie([0, 0.5, 0, 0.5, 0])
            S[i+1] = randNum

        elif randNum == 3:
            randNum = nSidedDie([0, 0, 0.6, 0, 0.4])
            S[i+1] = nSidedDie([0, 0, 0.6, 0, 0.4])

        elif randNum == 4:
            randNum = nSidedDie([0, 0, 0, 0, 1])
            S[i+1] = randNum

    plt.yticks([0, 1, 2, 3, 4])
    plt.scatter(np.arange(len(S)), S, color='r', edgecolors='b')
    plt.xticks(np.arange(len(S)), np.arange(0, len(S)))
    plt.plot(S, 'b:')
    plt.ylabel('State')
    plt.xlabel('Step Number')
    plt.title('Simulated Run of Five State Absorbing Markov Chain')
    plt.show()
    
fiveStateAbsorbingMarkovChain(15)
   
#4 
def simulatedAbsorptionChain(N,n):
    S = np.zeros(n)
    
    zeroCounter = 0
    fourCounter = 0

    for j in range(N):
        currentState = nSidedDie([0, 0, 1, 0, 0])
        S[0] = currentState
        for i in range(n-1):

            if currentState == 0:
                currentState = nSidedDie([1, 0, 0, 0, 0])
                S[i+1] = currentState

            elif currentState == 1:
                currentState = nSidedDie([0.3, 0, 0.7, 0, 0])
                S[i+1] = currentState

            elif currentState == 2:
                currentState = nSidedDie([0, 0.5, 0, 0.5, 0])
                S[i+1] = currentState

            elif currentState == 3:
                currentState = nSidedDie([0, 0, 0.6, 0, 0.4])
                S[i+1] = currentState

            elif currentState == 4:
                currentState = nSidedDie([0, 0, 0, 0, 1])
                S[i+1] = currentState

        if(S[len(S)-1] == 0):
            zeroCounter += 1
        if(S[len(S)-1] == 4):
            fourCounter += 1

    probAtZero = zeroCounter/N
    probAtFour = fourCounter/N
    
    print("Probability of Absorption for state 0 when starting at state 2: ", probAtZero)
    print("Probability of Absorption for state 4 when starting at state 2: ", probAtFour)

simulatedAbsorptionChain(10000, 15)