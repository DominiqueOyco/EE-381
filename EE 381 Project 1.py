#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 17:56:17 2019

@author: Pir8
"""
import math
import random
import numpy as np
import matplotlib.pyplot as plt      

#1
#histogram function counts how many times we got 3 rolls, 2 rolls, etc 
p = [0.10, 0.15, 0.20, 0.05, 0.30, 0.10, 0.10]
n = random.randint(1, 7)    
cs = np.cumsum(p)      
cp = np.append(0,cs)  
N = 10000
def nSidedDie(p):
    global d 
    r = np.random.rand() 
    for j in range(0,n):     
        if r>cp[j] and r<=cp[j+1]:      
            d = j+1   
    return d
r = nSidedDie(p)
print(r)

s = np.zeros((N,1))

for j in range(0, N):
    r = nSidedDie(p)     
    s[j]=r 
print(s)

 # Plotting             
b = range(1, n+2)     
sb = np.size(b)     
h1, bin_edges = np.histogram(s,bins=b)     
b1 = bin_edges[0: sb-1]     
plt.close('all')     
prob = h1/N     

# Graph labels   
plt.stem(b1, prob)    
plt.title('PMF for an unfair n-sided die')     
plt.xlabel('Number on the face of the die')     
plt.ylabel('Probability')       
plt.xticks(b1)          
#       

#2
def successDice7(N):
    S = []
    for i in range (0,N):        
        diceSum = 0  
        h = 0 #number of roll where roll = 7
        while(diceSum != 7):
            dice1 = random.randint(1, 7)
            dice2 = random.randint(1, 7)
            diceSum = dice1 + dice2
            h+=1
            if(diceSum == 7):
                S.append(h)
           
    b=range(1,40) ; sb=np.size(b)     
    h1, bin_edges = np.histogram(S,bins=b)     
    b1=bin_edges[0:sb-1]     
    plt.close('all')
    prob = h1/N

    #plotting    
    plt.stem(b1, prob)     
    plt.title('Stem plot - Sum of Two Dice that results in 7')     
    plt.xlabel('Number of rolls where sum of two dice is 7')     
    plt.ylabel('Probability')        

successDice7(100000)
            
#3
#return the number of heads    
def singleExperimentCoinToss(n):
    coins = np.random.randint(0, 2, n)
    heads = np.sum(coins)
    return heads

def nExperimentCoinToss(N):
    n = 100
    successCounter = []
    
    for l in range(0, N):    
        heads = singleExperimentCoinToss(n)
        k = heads                   #the number of success
        
        if k == 50:
            successCounter.append(k) 
    
    z = len(successCounter)     #the number of success after running N times              
    
    probSuccess = z / N
    
    print("The total number of successful attempts after running", N, 
    "times is", z)
    
    print("The probabilty of success for tossing", n, "coins", N, "times is:", 
          probSuccess)

h = singleExperimentCoinToss(100)
print(h)
nExperimentCoinToss(100000)

#4
k = 7
m = 80000
n = math.pow(26, 4)

#Single Experiment hackerlist of size m
def mSizeHacker(N):
    hackerList = []
    for x in range(0, m):
        pw = random.randint(0, n)
        hackerList.append(pw)
    
    mSuccessCounter = 0       
    for v in range(0, N):    
        myPass = random.randint(0, n) 
        if myPass in hackerList:
            mSuccessCounter += 1
    
    probMSuccess = mSuccessCounter / N
    return probMSuccess
    
#Single Experiment hackerlist of size k*m
def kmSizeHacker(N):        
    bigHackerList = []
    kmSuccessCounter = 0
    for y in range(0, k*m):
        pw = random.randint(0, n)
        bigHackerList.append(pw)

    for w in range(0, N):
        myPass = random.randint(0, n)
        if myPass in bigHackerList:
            kmSuccessCounter += 1

    probKmSuccess = kmSuccessCounter / N
    return probKmSuccess   
 
newM = 45000  
def approxNumber(N):    
    bigHackerList = []
    kmSuccessCounter = 0
    for z in range(0, k*newM):
        pw = random.randint(0, n)
        bigHackerList.append(pw)
    
    for u in range(0, N):
        myPass = random.randint(0, n)
        if myPass in bigHackerList:
            kmSuccessCounter += 1

    probKmSuccess = kmSuccessCounter / N
    return probKmSuccess
    
print("The probability that one password in the hackerlist is yours is: ",
      mSizeHacker(1000))   
    
print("The probability that one password in the bigger hackerlist is yours is: ",
          kmSizeHacker(1000))  
    
print("The Approximate Number of Words in the list is: ", approxNumber(1000), "if m is", newM)   
    

    
    
    




            



















#def singleExperimentCoinToss(n):
#    heads = coinTosser(n)
#    tails = n - heads
#    k = heads                   #the number of success
#    
#    if k == 50:
#        print("SUCCESS!")
#    else:
#        print("FAILURE!")
#    
#    p = heads/n                 #the probability of getting heads
#    q = tails/n                 #the probability of getting tails
#    
#    print("Heads: ", heads, "\t Tails:  ", tails)
#    #prob(50 in 100) = combination of n and k * p^k * q^k
#    probKinN = nCr(n, k) * math.pow(p, k) * math.pow(q, n-k)
#    print("Number of Successes: ", k)
#    print("Probability of Success: ", probKinN)



          
## 
#n=3 
#p=np.array([0.3, 0.6, 0.1]) 
#cs=np.cumsum(p)  
#cp=np.append(0,cs) 
#r= np.random.rand() 
#for k in range(0,n):     
#    if r>cp[k] and r<=cp[k+1]:            
#        d=k+1         
#p=np.array([0.3, 0.6, 0.1]) 
#
#def ThreeSidedDie(p):     
#    N=10000     
#    s=np.zeros((N,1))     
#    n=3     
#    #p=np.array([0.3, 0.6, 0.1])     
#    cs=np.cumsum(p)      
#    cp=np.append(0,cs)     
#    #     
#    for j in range(0,N):         
#        r=np.random.rand()         
#        for k in range(0,n):             
#            if r>cp[k] and r<=cp[k+1]:                    
#                d=k+1         
#                s[j]=d  
#
#    # Plotting             
#    b = range(0, n+3)     
#    sb = np.size(b)     
#    h1, bin_edges = np.histogram(s,bins=b)     
#    b1 = bin_edges[0: sb-1]     
#    plt.close('all')     
#    prob = h1/N     
#    plt.stem(b1, prob)  
#    
#    # Graph labels     
#    plt.title('PMF for an unfair 3-sided die')     
#    plt.xlabel('Number on the face of the die')     
#    plt.ylabel('Probability')       
#    plt.xticks(b1)          
#    # 
#       
#ThreeSidedDie(p)
    
#randomizedList = []
#
#for i in range(0, 6):
#    n = random.randint(1, 6)
#    randomizedList.append(n)
#
#print(randomizedList)
#print(randomizedList[3])
#
#n = random.randint(1,7)
#p = [0.10, 0.15, 0.20, 0.05, 0.30, 0.10, 0.10]
#cs = np.cumsum(p)
#cp = np.append(0,cs) 
#r = np.random.rand() 
#
#for k in range(0,n):     
#    if r > cp[k] and r <= cp[k+1]:            
#        d = k+1        
#  


#def multipleCoinToss(N):    
#    successCounter = []
#    failureCounter = []
##    counter = 0
#    for l in range(0, N):
#        y = 100
#        heads = coinTosser(y)
#        tails = n - heads
#        k = heads                       #the number of success
#        
#        if k == 50:
#            successCounter.append(k)    
#        else:
#            failureCounter.append(k)
#        
#        z = len(successCounter)     #the number of success after running N times
#        a = len(successCounter)/N      #the probability of getting heads
#        b = len(failureCounter)/N      #the probability of getting tails               
#    
#    probAinN = nCr(math.pow(10, 7), k) * math.pow(a, k) * math.pow(b, math.pow(10, 7) - k)
#    
#    print("The total number of successful attempts after running", N, 
#    "times is", z)
#    
#    print("The probabilty of success for tossing the coin", N, "times is:", 
#          probAinN)
    
#def singleExperimentCoinToss(n, k):  
#    coins = np.random.randint(0, 2, 100)
#    heads = np.sum(coins)
#    tails = n - heads
#    
#    probHeads = heads/n
#    probTails = tails/n
#    
#    print(heads, "\n", tails, "\n", probHeads, "\n", probTails, "\n")
#    
#    probKinN = nCr(100, 50) * math.pow(probHeads, k) * math.pow(probTails, k)
#    return probKinN

#def singleExperimentCoinToss(N, k):
#    for l in range(0, N): 
#        tosses = 100
#        coins = np.random.randint(0, 2, tosses)
#        heads = np.sum(coins)  
#        tails = tosses - heads
#        
#        probHeads = heads/tosses
#        probTails = tails/tosses
#        
#        probKinN = nCr(100, 50) * math.pow(probHeads, k) * math.pow(probTails, k)
#        return probKinN

 
#def continuousCoinToss(N):    
#    successCounter = []
#    counter = 0
#    for l in range(0, N):
#        experiment = singleExperimentCoinToss(100, 50)  
#        successCounter.append(experiment)
#    
#    for m in successCounter:
#        if (m == 0.5):
#            counter += m
#           
#    probSuccess = counter/N
#    
#    print("The total number of successful attempts after running", N, 
#    "times is", counter)
#    
#    print("The probabilty of success for tossing the coin", N, "times is:", 
#          probSuccess)
    
#tosser = singleExperimentCoinToss(100)
#print(tosser)
#continuousCoinToss(100000)        
        
        
        