#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 18:11:50 2019

@author: Pir8
"""
import random
import numpy as np

#1
#RUN 100,000 TIMES AND COUNT THE NUMBER OF ERRORS. DIVIDE THE ERROR BY N

#Generate Message
p0 = 0.6
e0 = 0.05
e1 = 0.03
N = 100000 
R = np.zeros((N,1)) #Received bit
S = np.zeros((N,1)) #Transmitted bit
badTransmissions = np.zeros((N,1))

for j in range(0,N): 
    #Transmit Bit
    M = np.random.rand() #returns a random number from 0 to 1
    if(M <= p0): 
        S[j] = 0        
    else:
        S[j] = 1
   
    #Receive Bit
    T = np.random.rand()  
    if(S[j] == 0 and T > e0):
        R[j] = 0    
    elif(S[j] == 0 and T <= e0):
        R[j] = 1
    elif(S[j] == 1 and T <= e1):
        R[j] = 0       
    elif(S[j] == 1 and T > e1):
        R[j] = 1
        
    #check if received bit = transmitted bit
    if(R[j] == S[j]): #if R=S, the experiment is a success
        badTransmissions[j] = 0
    elif(R[j] != S[j]): #if R!=S, the experiment is a failure
        badTransmissions[j] = 1
       
probFailTransmissions = np.sum(badTransmissions)/N
print("The probability of transmission error is: ", probFailTransmissions)

#2
#RUN 100,000 TIMES AND DIVIDE THE NUMBER OF SUCCESS BY THE SUM OF S
#Generate Message
p0 = 0.6
e0 = 0.05
e1 = 0.03
N = 100000 
R = np.zeros((N,1)) #Received bit
S = np.zeros((N,1)) #Transmitted bit
successCounter = np.zeros((N,1))
for k in range(0,N): 
    #Transmit Bit
    M = np.random.rand()
    if(M <= p0): 
        S[k] = 0        
    else:
        S[k] = 1
        
    
    #Receive Bit
    T = np.random.rand()  
    if(S[k] == 0 and T > e0):
        R[k] = 0
    if(S[k] == 0 and T <= e0):
        R[k] = 1
    elif(S[k] == 1 and T <= e1):
        R[k] = 0       
    elif(S[k] == 1 and T > e1):
        R[k] = 1
        
    #check if received bit = transmitted bit
    if(R[k] == 1 and S[k] == 1): #if R = 1 & S = 1, the experiment is a success
        successCounter[k] = 1
    else: #if R = 1 and S = 0 or vice versa, the experiment is a failure
        successCounter[k] = 0

r1s1Prob = np.sum(successCounter)/np.sum(S)
print("The conditional probability when P(R=1|S=1) is: ", r1s1Prob)

#3
#RUN 100,000 TIMES AND DIVIDE THE NUMBER OF SUCCESS BY THE SUM OF R

#Generate Message
p0 = 0.6
e0 = 0.05
e1 = 0.03
N = 100000 
R = np.zeros((N,1)) #Received bit
S = np.zeros((N,1)) #Transmitted bit
successCounter = np.zeros((N,1))
for l in range(0,N): 
    #Transmit Bit
    M = np.random.rand()
    if(M <= p0): #if M <= 0.6, append 0 to S[]
        S[l] = 0        
    else:
        S[l] = 1
        
    #Receive Bit
    T = np.random.rand()  
    if(S[l] == 0 and T > e0):
        R[l] = 0
    if(S[l] == 0 and T <= e0):
        R[l] = 1
    elif(S[l] == 1 and T <= e1):
        R[l] = 0       
    elif(S[l] == 1 and T > e1):
        R[l] = 1
        
    #check if received bit = transmitted bit
    if(S[l] == 1 and R[l] == 1): #if S = 1 & R = 1, the experiment is a success
        successCounter[l] = 1
    else: #if S = 1 and R = 0 or vice versa, the experiment is a failure
        successCounter[l] = 0
    
s1r1Prob = np.sum(successCounter)/np.sum(R)
print("The conditional probability when P(S=1|R=1) is: ", s1r1Prob)

#4
#RUN 100,000 TIMES AND COUNT THE NUMBER OF ERRORS. DIVIDE THE ERROR BY N
#Generate Message
p0 = 0.6
e0 = 0.05
e1 = 0.03
N = 100000 
R1 = np.zeros((N,1)) #Received bit
R2 = np.zeros((N,1)) #Received bit
R3 = np.zeros((N,1)) #Received bit
D = np.zeros((N,1)) #Decoded Bits
S = np.zeros((N,1)) #Transmitted bit

for x in range(0,N): 
    #Transmit Bit
    M = np.random.rand()
    if(M <= p0): 
        S[j] = 0        
    else:
        S[j] = 1
        
    T1 = np.random.rand()  
    T2 = np.random.rand()
    T3 = np.random.rand()
    
    #Receiving bit for T1
    if(S[x] == 0 and T1 > e0):
        R1[x] = 0 #1-e0  
    elif(S[x] == 0 and T1 <= e0):
        R1[x] = 1 #P(ERROR) = e0
    elif(S[x] == 1 and T1 <= e1):
        R1[x] = 0 #P(ERROR) = e1       
    elif(S[x] == 1 and T1 > e1):
        R1[x] = 1 #1-e1
        
    #Receiving bit for T2
    if(S[x] == 0 and T2 > e0):
        R2[x] = 0 #1-e0  
    elif(S[x] == 0 and T2 <= e0):
        R2[x] = 1 #P(ERROR) = e0
    elif(S[x] == 1 and T2 <= e1):
        R2[x] = 0 #P(ERROR) = e1       
    elif(S[x] == 1 and T2 > e1):
        R2[x] = 1 #1-e1

    #Receiving bit for T2
    if(S[x] == 0 and T3 > e0):
        R3[x] = 0 #1-e0  
    elif(S[x] == 0 and T3 <= e0):
        R3[x] = 1 #P(ERROR) = e0
    elif(S[x] == 1 and T3 <= e1):
        R3[x] = 0 #P(ERROR) = e1       
    elif(S[x] == 1 and T3 > e1):
        R3[x] = 1 #1-e1
        
    if(S[x] == 0):
        if(R1[x] + R2[x] + R3[x] <= 1):
            D[x] = 0
        else:
            D[x] = 1
    elif(S[x] == 1):
        if(R1[x] + R2[x] + R3[x] >= 2):
            D[x] = 0
        else:
            D[x] = 1            
    
probFailTransmissions = np.sum(D)/N
print("The probability of transmission error is: ", probFailTransmissions)



















##Generate Message
#N = 100000
##n = random.randint(0,7)  
#p = [p0, 1-p0] 
#q = [1-e0, e0]
#R = np.zeros((N,1)) #Received bit
#S = np.zeros((N,1)) #Transmitted bit
#badTransmissions = np.zeros((N,1))
##cs = np.cumsum(p)      
##cp = np.append(0,cs) 
##
##def nSidedDie(p):
##    global d 
##    r = np.random.rand() 
##    for i in range(0,n):     
##        if r>cp[i] and r<=cp[i+1]:      
##            d = i   
##    return d
#
#for j in range(0,N): 
#    #Transmit Bit
##    M = nSidedDie(p)
##    if(M <= p0): 
##        s = 0
##        S[j] = s        
##    else:
##        s = 1
#    M = random.random()
#    if(M <= p0): 
#        S[j] = 0        
#    else:
#        S[j] = 1
#    #Receive Bit
#    T = random.randint(0,1)   
#    if(S[j] == 0 and T <= e0):
#        r = 1
#    elif(S[j] == 0 and T > e0):
#        r = 0
#    elif(S[j] == 1 and T <= e1):
#        r = 0       
#    elif(S[j] == 1 and T > e1):
#        r = 1
#    R[j] = r
#    #check if received bit = transmitted bit
#    if(R[j] == S[j]): #if R=S, the experiment is a success
#        badTransmissions[j] = 0
#    elif(R[j] != S[j]): #if R!=S, the experiment is a failure
#        badTransmissions[j] = 1
#    
#print(sum(badTransmissions))     
#probFailTransmissions = np.sum(badTransmissions)/N
#print("The probability of transmission error is: ", probFailTransmissions)