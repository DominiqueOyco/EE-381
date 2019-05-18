import math
import numpy as np
import matplotlib.pyplot as plt  

N = 10000
p = [0.2, 0.1, 0.15, 0.3, 0.2, 0.05]  
c = [1, 2, 3, 4, 5, 6]
X = np.zeros((N,1)) #Stores the total value of success of each rolls for N = 10000 times

#1. 
#NOTES: COUNT THE NUMBER OF SUCCESSES IN n = 1000 TRIALS THEN STORE IT IN X
#       SUCCESS IF DEFINED IF DICE1 = 1, DICE2 = 2 AND DICE3 = 3 
def bernoulli():   
    for i in range(0, N):  
        s = 0 #counts the success
        n = 1000
        d1 = np.random.choice(c, n, p) #generates a random number from 1 to 6 on dice 1
        d2 = np.random.choice(c, n, p) #generates a random number from 1 to 6 on dice 2 
        d3 = np.random.choice(c, n, p) #generates a random number from 1 to 6 on dice 3 
        for j in range(0, n):
            if(d1[j] == 1):
                if(d2[j] == 2):
                    if(d3[j] == 3):
                        s+=1 
        X[i] = s
        
    # Plotting             
    b = range(0, 15)     
    sb = np.size(b)     
    h1, bin_edges = np.histogram(X,bins=b)     
    b1 = bin_edges[0: sb-1]     
    plt.close('all')     
    prob = h1/N     
    
    #plotting   
    plt.stem(b1, prob)      
    plt.title('Bernoulli Trials: PMF Experimental Results')     
    plt.xlabel('Number of successes in n = 1000 trials')     
    plt.ylabel('Probability')               
bernoulli()

#2 Find the probability of success in a single roll of three dice using Binomial Distribution
#BINOMIAL DISTRIBUTION FORMULA P(X) = nCr(a,b) * p^r * q^(n-r) 
#X is the number of success in problem 1
#q = 1-p

#THIS FUNCTION IS THE COMBINATION FORMULA nCr = n!/r!(n-r)!
def nCr(n,r): 
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def binomialDist():  
    n = 1000
    p = math.pow((1/6), 3)
    q = math.pow((5/6), 3) #q = 1 - p (1-1/6 = 5/6)
    B = np.zeros((15,1)) #stores the probability to be plotted on the PMF
    
    for k in range(0, 15):
        t = n - k
        B[k] = nCr(n, k) * math.pow(p, k) * math.pow(q, t) 
      
    #plotting      
    plt.stem(B) 
    
    #plotting       
    plt.title('Bernoulli Trials: Binomial Formula')     
    plt.xlabel('Number of successes in n = 1000 trials')     
    plt.ylabel('Probability') 

binomialDist()

#3 Use Poisson Approximation to approximate the probability of success in n trials as 
#as an alternative to the Binomial formula
#POISSON APPROXIMATION FORMULA P(X) = (lambda^v)(e^-lambda) / l!
#X is the number of success in problem 1

def poissonApprox(): 
    n = 1000
    p = math.pow((1/6), 3)
    approx = n * p #lambda
    P = np.zeros((15,1)) #stores the approximations to be plotted on the PMF
    
    for l in range(0, 15):
        v = l - 1
        P[l] = (math.exp(-approx) * math.pow(approx, v)) / math.factorial(l) #poison approx formila
            
    #plotting      
    plt.stem(P)
    
    #plotting       
    plt.title('Bernoulli Trials PMF: Poisson Approximation')     
    plt.xlabel('Number of successes in n = 1000 trials')     
    plt.ylabel('Probability') 
    
poissonApprox()


