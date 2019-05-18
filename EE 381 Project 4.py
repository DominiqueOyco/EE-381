#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 18:55:01 2019

@author: rain
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt  

#1.1 SIMULATE A UNIFORM RANDOM VARIABLE
def uniformRandVar():
    #generates the values of the random variable X
    a = 1
    b = 4
    n = 10000
    X = np.random.uniform(a,b,n)
      
    #Create bins and histogram
    nbins=20; 
    edgecolor='w';
    #
    bins=[float(x) for x in np.linspace(a, b, nbins+1)]
    h1, bin_edges = np.histogram(X, bins, density=True)
    be1=bin_edges[0:np.size(bin_edges)-1]
    be2=bin_edges[1:np.size(bin_edges)]
    b1=(be1+be2)/2
    barwidth=b1[1]-b1[0]
    plt.close('all')
    
    #plot the bar graph
    fig1=plt.figure(1)
    plt.bar(b1,h1,width=barwidth,edgecolor=edgecolor)
    
    #PLOTS THE UNIFORM PDF
    def UnifPDF(a,b,x):
         f=(1/abs(b-a))*np.ones(np.size(x))
         return f
     
    f=UnifPDF(a,b,b1)
    plt.plot(b1,f,'r')
    plt.title('Bar Graph: Probability Density Function of a Uniform Random Variable')     
    plt.xlabel('X')     
    plt.ylabel('P(X)') 
    
    #calculates the mean and standard deviation
    mu_x = np.mean(X)
    sig_x = np.std(X)
    
    print("The mean is: ", mu_x)
    print("The standard deviation is: ", sig_x)

uniformRandVar()

#1.2 
def exponentialRandVar():
    #generates the values of the exponential random variable T
    n = 10000
    beta = 40
    T = np.random.exponential(beta,n)
          
    #Create bins and histogram
    nbins=20; 
    edgecolor='w';
    
    #
    bins=[float(t) for t in np.linspace(0, 100, nbins+1)]
    h1, bin_edges = np.histogram(T, bins, density=True)
    be1=bin_edges[0:np.size(bin_edges)-1]
    be2=bin_edges[1:np.size(bin_edges)]
    b1=(be1+be2)/2
    barwidth=b1[1]-b1[0]
    plt.close('all')
    
    #plot the bar graph
    fig1=plt.figure(1)
    plt.bar(b1,h1,width=barwidth,edgecolor=edgecolor)
    
    def expPDF(beta,t):
         f =((1/abs(beta))* np.exp(-(1/abs(beta))*t)) * np.ones(np.size(t))
         return f
     
    #PLOT THE EXPONENTIAL PDF 
    f=expPDF(beta,b1)
    plt.plot(b1,f,'r')
    plt.title('Bar Graph: Probability Density Function of a  Exponential Random Variable')     
    plt.xlabel('T')     
    plt.ylabel('P(T)') 
    
    #calculates the mean and standard deviation
    mu_t = np.mean(T)
    sig_t = np.std(T)
    
    print("The mean is: ", mu_t)
    print("The standard deviation is: ", sig_t)

exponentialRandVar()

#1.3 
def normalRandVar():
    #generates the values of the random variable X
    n = 10000
    mu = 2.5
    sigma = 0.75
    X = np.random.normal(mu,sigma,n)
          
    #Create bins and histogram
    nbins=10; 
    edgecolor='w';
    
    #
    bins=[float(x) for x in np.linspace(0, 6, nbins+1)]
    h1, bin_edges = np.histogram(X, bins, density=True)
    be1=bin_edges[0:np.size(bin_edges)-1]
    be2=bin_edges[1:np.size(bin_edges)]
    b1=(be1+be2)/2
    barwidth=b1[1]-b1[0]
    plt.close('all')
    
    #plot the bar graph
    fig1=plt.figure(1)
    plt.bar(b1,h1,width=barwidth,edgecolor=edgecolor)
    
    def normalPDF(mu,sigma,x):
         f = (1/(sigma * np.sqrt(2*np.pi)))*np.exp(-np.power((x-mu),2)/(2*np.power(sigma,2)))*np.ones(np.size(x))
         return f
     
    #PLOT THE NORMAL PDF 
    f=normalPDF(mu,sigma,b1)
    plt.plot(b1,f,'r')
    plt.title('Bar Graph: Probability Density Function of a a Normal Random Variable')     
    plt.xlabel('X')     
    plt.ylabel('P(X)') 
    
    #calculates the mean and standard deviation
    mu_x = np.mean(X)
    sig_x = np.std(X)
    
    print("The mean is: ", mu_x)
    print("The standard deviation is: ", sig_x)

normalRandVar()

#2.1 - if n is equal to 1
def centLimThm():
    n = 1
    N = 10000
    a = 1
    b = 4
    mu_x=(a+b)/2
    sig_x=np.sqrt((b-a)**2/12)  
    X=np.zeros((N,1)) 
    
    for k in range(0,N):
        x=np.random.uniform(a,b,n)
        W=np.sum(x) #The thickness of the stack
        X[k]=W
        
    # Create bins and histogram
    nbins=20 # Number of bins 
    edgecolor='w'
    # Color separating bars in the bargraph # 
    bins=[float(x) for x in np.linspace(n*a, n*b,nbins+1)]
    h1, bin_edges = np.histogram(X,bins,density=True)
    # Define points on the horizontal axis
    be1=bin_edges[0:np.size(bin_edges)-1]
    be2=bin_edges[1:np.size(bin_edges)]
    b1=(be1+be2)/2
    barwidth=b1[1]-b1[0]
    # Width of bars in the bargraph
    plt.close('all')
    
    # PLOT THE BAR GRAPH
    fig1=plt.figure(1)
    plt.bar(b1,h1, width=barwidth, edgecolor=edgecolor)
    plt.title('PDF of book stack height and comparison with Gaussian')     
    plt.xlabel('Book stack height for n=1 books')     
    plt.ylabel('PDF') 
    
    #PLOT THE GAUSSIAN FUNCTION
    def gaussian(mu,sig,z):
        f=np.exp(-(z-mu)**2/(2*sig**2))/(sig*np.sqrt(2*np.pi))
        return f
    
    f=gaussian(mu_x*n,sig_x*np.sqrt(n),b1)
    plt.plot(b1,f,'r')
    
    #calculates the mean and standard deviation
    expMu_x = np.mean(X)
    sigMu_x = np.std(X)
    
    print("The experimental mean is: ", expMu_x)
    print("The experimental standard deviation is: ", sigMu_x)
    
centLimThm()

#2.2 - if n is equal to 5
def centLimThm2():
    n = 5
    N = 10000
    a = 1
    b = 4
    mu_x=(a+b)/2
    sig_x=np.sqrt((b-a)**2/12)  
    X=np.zeros((N,1)) 
    
    for k in range(0,N):
        x=np.random.uniform(a,b,n)
        w=np.sum(x)
        X[k]=w
        
    # Create bins and histogram
    nbins=30 # Number of bins 
    edgecolor='w'
    # Color separating bars in the bargraph # 
    bins=[float(x) for x in np.linspace(n*a, n*b,nbins+1)]
    h1, bin_edges = np.histogram(X,bins,density=True)
    # Define points on the horizontal axis
    be1=bin_edges[0:np.size(bin_edges)-1]
    be2=bin_edges[1:np.size(bin_edges)]
    b1=(be1+be2)/2
    barwidth=b1[1]-b1[0]
    # Width of bars in the bargraph
    plt.close('all')
    
    # PLOT THE BAR GRAPH
    fig2=plt.figure(2)
    plt.bar(b1,h1, width=barwidth, edgecolor=edgecolor)
    plt.title('PDF of book stack height and comparison with Gaussian')     
    plt.xlabel('Book stack height for n=5 books')     
    plt.ylabel('PDF') 
    
    #PLOT THE GAUSSIAN FUNCTION
    def gaussian(mu,sig,z):
        f=np.exp(-(z-mu)**2/(2*sig**2))/(sig*np.sqrt(2*np.pi))
        return f
    
    f=gaussian(mu_x*n,sig_x*np.sqrt(n),b1)
    plt.plot(b1,f,'r')
    
    #calculates the mean and standard deviation
    expMu_x = np.mean(X)
    sigMu_x = np.std(X)
    
    print("The experimental mean is: ", expMu_x)
    print("The experimental standard deviation is: ", sigMu_x)
    
centLimThm2() 
    
#2.3 - if n is equal to 15
def centLimThm3():
    n = 15
    a = 1
    b = 4
    N = 10000
    mu_x=(a+b)/2
    sig_x=np.sqrt((b-a)**2/12)  
    X=np.zeros((N,1)) 
    
    for k in range(0,N):
        x=np.random.uniform(a,b,n)
        w=np.sum(x)
        X[k]=w
        
    # Create bins and histogram
    nbins=30 # Number of bins 
    edgecolor='w'
    # Color separating bars in the bargraph # 
    bins=[float(x) for x in np.linspace(n*a, n*b,nbins+1)]
    h1, bin_edges = np.histogram(X,bins,density=True)
    # Define points on the horizontal axis
    be1=bin_edges[0:np.size(bin_edges)-1]
    be2=bin_edges[1:np.size(bin_edges)]
    b1=(be1+be2)/2
    barwidth=b1[1]-b1[0]
    # Width of bars in the bargraph
    plt.close('all')
    
    # PLOT THE BAR GRAPH
    fig3=plt.figure(3)
    plt.bar(b1,h1, width=barwidth, edgecolor=edgecolor)
    plt.title('PDF of book stack height and comparison with Gaussian')     
    plt.xlabel('Book stack height for n=15 books')     
    plt.ylabel('PDF') 
    
    #PLOT THE GAUSSIAN FUNCTION
    def gaussian(mu,sig,z):
        f=np.exp(-(z-mu)**2/(2*sig**2))/(sig*np.sqrt(2*np.pi))
        return f
    
    f=gaussian(mu_x*n,sig_x*np.sqrt(n),b1)
    plt.plot(b1,f,'r')    
    
    #calculates the mean and standard deviation
    expMu_x = np.mean(X)
    sigMu_x = np.std(X)
    
    print("The experimental mean is: ", expMu_x)
    print("The experimental standard deviation is: ", sigMu_x)
   
centLimThm3()

#3
def distOfSumOfRandVar():
    n = 24
    N = 10000
    beta = 40
    C=np.zeros((N,1)) 
    
    for i in range(0,N):
        T = np.random.exponential(beta,n)
        w=np.sum(T)
        C[i]=w
        
    # Create bins and histogram
    nbins=30 # Number of bins 
    edgecolor='w'
    
    # Color separating bars in the bargraph # 
    bins=[float(t) for t in np.linspace(0, 50*beta, nbins+1)]
    h1, bin_edges = np.histogram(C, bins, density=True)
    # Define points on the horizontal axis
    be1=bin_edges[0:np.size(bin_edges)-1]
    be2=bin_edges[1:np.size(bin_edges)]
    b1=(be1+be2)/2
    barwidth=b1[1]-b1[0]
    # Width of bars in the bargraph
    plt.close('all')
    
    # PLOT THE BAR GRAPH
    fig1=plt.figure(1)
    plt.bar(b1,h1, width=barwidth, edgecolor=edgecolor)
    plt.title('PDF of battery lifetime')     
    plt.xlabel('24 Batteries in a carton')     
    plt.ylabel('PDF') 
    
    #PLOT THE GAUSSIAN FUNCTION
    def gaussian(mu,sig,z):
        f=np.exp(-(z-mu)**2/(2*sig**2))/(sig*np.sqrt(2*np.pi))
        return f
    
    f=gaussian(beta*n,beta*np.sqrt(n),b1)
    plt.plot(b1,f,'r')    
    
    #MAKING THE CDF    
    E = np.cumsum(f*barwidth)
       
    # PLOT THE CDF
    fig2=plt.figure(2)
    plt.bar(b1,E, width=barwidth, edgecolor=edgecolor)
    plt.title('CDF of battery lifetime')     
    plt.xlabel('24 Batteries in a carton')     
    plt.ylabel('CDF') 
    
    plt.plot(b1,E,'r--')   
    
distOfSumOfRandVar()






















































#def distOfSumOfRandVar():
#    #generates the values of the random variable X
#    N = 10000
#    n = 24
#    beta = 40
#    C = np.zeros((N,1))
#    
#    for i in range(0,N):  
#        T = np.random.exponential(beta,n)
#        c = np.sum(T)
#        C[i] = c 
#        
#    #Create bins and histogram
#    nbins=20; 
#    edgecolor='w';
#    
#    #
#    bins=[float(t) for t in np.linspace(0, 100, nbins+1)]
#    h1, bin_edges = np.histogram(C, bins, density=True)
#    be1=bin_edges[0:np.size(bin_edges)-1]
#    be2=bin_edges[1:np.size(bin_edges)]
#    b1=(be1+be2)/2
#    barwidth=b1[1]-b1[0]
#    plt.close('all')
#    
#    #calculates the mean and standard deviation
#    mu_t = np.mean(T)
#    sig_t = np.std(T)
#    
##    #calculates the mean and standard deviation of c
##    mu_c = n * np.mean(T)
##    sig_c = np.std(T) * math.sqrt(n)
#    
#    #plot the 1st bar graph
#    fig1=plt.figure(1)
#    plt.bar(b1,h1,width=barwidth,edgecolor=edgecolor)
#    
#    def expPDF(beta,t):
#         f =((1/abs(beta))* np.exp(-(1/abs(beta))*t)) * np.ones(np.size(t))
#         return f
#     
#    def normalPDF(mu,sigma,x):
#         f = (1/(sigma * np.sqrt(2*np.pi)))*np.exp(-np.power((x-mu),2)/(2*np.power(sigma,2)))*np.ones(np.size(x))
#         return f
#     
#    #PLOT THE EXPONENTIAL PDF 
#    ePdf=expPDF(beta,b1)
#    plt.plot(b1,ePdf,'r')
#    plt.title('Bar Graph: Probability Density Function of a Random Variable')     
#    plt.xlabel('b-axis')     
#    plt.ylabel('a-axis') 
#    
##    #PLOT THE NORMAL PDF 
##    nPdf=normalPDF(mu_t,sig_t,b1)
##    plt.plot(b1,nPdf,'y')
##    plt.title('Bar Graph: Probability Density Function of a Random Variable')     
##    plt.xlabel('b-axis')     
##    plt.ylabel('a-axis') 
#    
#    #plot the 2nd bar graph
#    fig2=plt.figure(2)
#    plt.bar(b1,h1,width=barwidth,edgecolor=edgecolor)
#    
#    #PLOT THE CDF 
#    g=expPDF(beta,b1)
#    cdf = np.cumsum(g*barwidth)
#    plt.plot(b1,cdf,'r')
#    plt.title('Experimental CDF: A Lifetime of a Carton')     
#    plt.xlabel('b-axis')     
#    plt.ylabel('a-axis') 
#    
#    
#    print("The mean is: ", mu_t)
#    print("The standard deviation is: ", sig_t)
#
#distOfSumOfRandVar()

