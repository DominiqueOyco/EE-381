#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:34:30 2019

@author: rain
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt  

#1 Effect of sample size on confidence intervals
def confidenceIntervals():
    N = 1500000
    mu_x = 55
    sigma_x = 5
    n = 1
    M = [] #stores the mean for each iteration
    S = [] #stores the standard deviation for each iteration
    I = [] #stores the n for each iteration
    
    while(n != 201):
        B = np.random.normal(mu_x, sigma_x, N)
        X = B[random.sample(range(N), n)]
        
        x_bar = np.mean(X)
        s = np.std(X, ddof=1)
        
        M.append(x_bar) #store each mean inside list M
        S.append(s)     #store each standard deviation inside list S
        I.append(n)     #store each iteration inside list I
        n+=1            #increment until n reaches 200

    # plotting the graph and the scatter graph for 95%
    fig1=plt.figure(1)
    plt.title('Sample Means and 95% confidence intervals')
    plt.xlabel('Sample Size')
    plt.ylabel('x_bar')
    plt.scatter(I, M, marker='x')
    topCurve = ninetyFiveConfidenceSample(mu_x, sigma_x, I)
    bottomCurve = ninetyFiveConfidenceSample(mu_x, -1*sigma_x, I)
    plt.plot(I, topCurve, 'r--')
    plt.plot(I, bottomCurve, 'r--')
    plt.show()

    # plotting the graph and the scatter graph for 99%
    fig2=plt.figure(2)
    plt.title('Sample Means and 99% confidence intervals')
    plt.xlabel('Sample Size')
    plt.ylabel('x_bar')
    plt.scatter(I, M, marker='x')
    topCurve = ninetyNineConfidenceSample(mu_x, sigma_x, I)
    bottomCurve = ninetyNineConfidenceSample(mu_x, -1*sigma_x, I)
    plt.plot(I, topCurve, 'g--')
    plt.plot(I, bottomCurve, 'g--')
    plt.show()

def ninetyFiveConfidenceSample(mu, sd, n):
    X = []    
    for i in range(200):
        z = mu + 1.96 * (sd/math.sqrt(n[i]))
        X.append(z)
    return X

def ninetyNineConfidenceSample(mu, sd, n):
    X = []    
    for j in range(200):
        z = mu + 2.58 * (sd/math.sqrt((n[j])))
        X.append(z)

    return X
    
confidenceIntervals()


#2.1 95% confidence for normal distribution with n = 5
def ninetyFiveConfidenceNormal1():
    mu_x = 55
    sigma_x = 5
    M = 10000
    N = 1500000
    n = 5
    successCounter = []
    B = np.random.normal(mu_x, sigma_x, N)
    
    for i in range(0, M):
        X = B[random.sample(range(N), n)]
        x_bar = np.mean(X)
        s_hat = np.std(X, ddof=1)
        
        mu_lower = x_bar - 1.96 * (s_hat/math.sqrt(n))
        mu_upper = x_bar + 1.96 * (s_hat/math.sqrt(n))
    
        if(mu_x > mu_lower and mu_x < mu_upper):
            successCounter.append(i)

    successProb = len(successCounter)/M
    print("The percentage of success with 95% confidence using normal distribution n =", n, "is", successProb)

ninetyFiveConfidenceNormal1()

#2.2 95% confidence for normal distribution n = 40
def ninetyFiveConfidenceNormal2():
    mu_x = 55
    sigma_x = 5
    M = 10000
    N = 1500000
    n = 40
    successCounter = []
    B = np.random.normal(mu_x, sigma_x, N)
    
    for i in range(0, M):
        X = B[random.sample(range(N), n)]
        x_bar = np.mean(X)
        s_hat = np.std(X, ddof=1)
        
        mu_lower = x_bar - 1.96 * (s_hat/math.sqrt(n))
        mu_upper = x_bar + 1.96 * (s_hat/math.sqrt(n))
    
        if(mu_x > mu_lower and mu_x < mu_upper):
            successCounter.append(i)

    successProb = len(successCounter)/M
    print("The percentage of success with 95% confidence using normal distribution n =", n, "is", successProb)

ninetyFiveConfidenceNormal2()

#2.3 95% confidence for normal distribution with n = 120
def ninetyFiveConfidenceNormal3():
    mu_x = 55
    sigma_x = 5
    M = 10000
    N = 1500000
    n = 120
    successCounter = []
    B = np.random.normal(mu_x, sigma_x, N)
    
    for i in range(0, M):
        X = B[random.sample(range(N), n)]
        x_bar = np.mean(X)
        s_hat = np.std(X, ddof=1)
        
        mu_lower = x_bar - 1.96 * (s_hat/math.sqrt(n))
        mu_upper = x_bar + 1.96 * (s_hat/math.sqrt(n))
    
        if(mu_x > mu_lower and mu_x < mu_upper):
            successCounter.append(i)

    successProb = len(successCounter)/M
    print("The percentage of success with 95% confidence using normal distribution n =", n, "is", successProb)

ninetyFiveConfidenceNormal3()

#2.4 99% confidence for normal distribution with n = 5
def ninetyNineConfidenceNormal1():
    mu_x = 55
    sigma_x = 5
    M = 10000
    N = 1500000
    n = 5
    successCounter = []
    B = np.random.normal(mu_x, sigma_x, N)
    
    for i in range(0, M):
        X = B[random.sample(range(N), n)]
        x_bar = np.mean(X)
        s_hat = np.std(X, ddof=1)
        
        mu_lower = x_bar - 2.58 * (s_hat/math.sqrt(n))
        mu_upper = x_bar + 2.58 * (s_hat/math.sqrt(n))
    
        if(mu_x > mu_lower and mu_x < mu_upper):
            successCounter.append(i)

    successProb = len(successCounter)/M
    print("The percentage of success with 99% confidence using normal distribution n =", n, "is", successProb)

ninetyNineConfidenceNormal1()

#2.5 99% confidence for normal distribution with n = 40
def ninetyNineConfidenceNormal2():
    mu_x = 55
    sigma_x = 5
    M = 10000
    N = 1500000
    n = 40
    successCounter = []
    B = np.random.normal(mu_x, sigma_x, N)
    
    for i in range(0, M):
        X = B[random.sample(range(N), n)]
        x_bar = np.mean(X)
        s_hat = np.std(X, ddof=1)
        
        mu_lower = x_bar - 2.58 * (s_hat/math.sqrt(n))
        mu_upper = x_bar + 2.58 * (s_hat/math.sqrt(n))
    
        if(mu_x > mu_lower and mu_x < mu_upper):
            successCounter.append(i)

    successProb = len(successCounter)/M
    print("The percentage of success with 99% confidence using normal distribution n =", n, "is", successProb)

ninetyNineConfidenceNormal2()

#2.6 99% confidence for normal distribution with n = 120
def ninetyNineConfidenceNormal3():
    mu_x = 55
    sigma_x = 5
    M = 10000
    N = 1500000
    n = 120
    successCounter = []
    B = np.random.normal(mu_x, sigma_x, N)
    
    for i in range(0, M):
        X = B[random.sample(range(N), n)]
        x_bar = np.mean(X)
        s_hat = np.std(X, ddof=1)
        
        mu_lower = x_bar - 2.58 * (s_hat/math.sqrt(n))
        mu_upper = x_bar + 2.58 * (s_hat/math.sqrt(n))
    
        if(mu_x > mu_lower and mu_x < mu_upper):
            successCounter.append(i)

    successProb = len(successCounter)/M
    print("The percentage of success with 99% confidence using normal distribution n =", n, "is", successProb)

ninetyNineConfidenceNormal3()

#2.7 95% confidence for student t with n = 5
def ninetyFiveConfidenceT1():
    mu_x = 55
    sigma_x = 5
    M = 10000
    N = 1500000
    n = 5
    successCounter = []
    B = np.random.normal(mu_x, sigma_x, N)
    
    for i in range(0, M):
        X = B[random.sample(range(N), n)]
        x_bar = np.mean(X)
        s_hat = np.std(X, ddof=1)
        
        mu_lower = x_bar - 2.78 * (s_hat/math.sqrt(n))
        mu_upper = x_bar + 2.78 * (s_hat/math.sqrt(n))
    
        if(mu_x > mu_lower and mu_x < mu_upper):
            successCounter.append(i)

    successProb = len(successCounter)/M
    print("The percentage of success with 95% confidence using t distribution n =", n, "is", successProb)

ninetyFiveConfidenceT1()

#2.8 95% confidence for student t with n = 40
def ninetyFiveConfidenceT2():
    mu_x = 55
    sigma_x = 5
    M = 10000
    N = 1500000
    n = 40
    successCounter = []
    B = np.random.normal(mu_x, sigma_x, N)
    
    for i in range(0, M):
        X = B[random.sample(range(N), n)]
        x_bar = np.mean(X)
        s_hat = np.std(X, ddof=1)
        
        mu_lower = x_bar - 2.02 * (s_hat/math.sqrt(n))
        mu_upper = x_bar + 2.02 * (s_hat/math.sqrt(n))
    
        if(mu_x > mu_lower and mu_x < mu_upper):
            successCounter.append(i)

    successProb = len(successCounter)/M
    print("The percentage of success with 95% confidence using t distribution n =", n, "is", successProb)

ninetyFiveConfidenceT2()

#2.9 95% confidence for student t with n = 120
def ninetyFiveConfidenceT3():
    mu_x = 55
    sigma_x = 5
    M = 10000
    N = 1500000
    n = 120
    successCounter = []
    B = np.random.normal(mu_x, sigma_x, N)
    
    for i in range(0, M):
        X = B[random.sample(range(N), n)]
        x_bar = np.mean(X)
        s_hat = np.std(X, ddof=1)
        
        mu_lower = x_bar - 1.98 * (s_hat/math.sqrt(n))
        mu_upper = x_bar + 1.98 * (s_hat/math.sqrt(n))
    
        if(mu_x > mu_lower and mu_x < mu_upper):
            successCounter.append(i)

    successProb = len(successCounter)/M
    print("The percentage of success with 95% confidence using t distribution n =", n, "is", successProb)

ninetyFiveConfidenceT3()

#2.10 99% confidence for student t with n = 5
def ninetyNineConfidenceT1():
    mu_x = 55
    sigma_x = 5
    M = 10000
    N = 1500000
    n = 5
    successCounter = []
    B = np.random.normal(mu_x, sigma_x, N)
    
    for i in range(0, M):
        X = B[random.sample(range(N), n)]
        x_bar = np.mean(X)
        s_hat = np.std(X, ddof=1)
        
        mu_lower = x_bar - 4.60 * (s_hat/math.sqrt(n))
        mu_upper = x_bar + 4.60 * (s_hat/math.sqrt(n))
    
        if(mu_x > mu_lower and mu_x < mu_upper):
            successCounter.append(i)

    successProb = len(successCounter)/M
    print("The percentage of success with 99% confidence using t distribution n =", n, "is", successProb)

ninetyNineConfidenceT1()

#2.11 99% confidence for student t with n = 40
def ninetyNineConfidenceT2():
    mu_x = 55
    sigma_x = 5
    M = 10000
    N = 1500000
    n = 40
    successCounter = []
    B = np.random.normal(mu_x, sigma_x, N)
    
    for i in range(0, M):
        X = B[random.sample(range(N), n)]
        x_bar = np.mean(X)
        s_hat = np.std(X, ddof=1)
        
        mu_lower = x_bar - 2.71 * (s_hat/math.sqrt(n))
        mu_upper = x_bar + 2.71 * (s_hat/math.sqrt(n))
    
        if(mu_x > mu_lower and mu_x < mu_upper):
            successCounter.append(i)

    successProb = len(successCounter)/M
    print("The percentage of success with 99% confidence using t distribution n =", n, "is", successProb)

ninetyNineConfidenceT2()

#2.12 99% confidence for student t with n = 120
def ninetyNineConfidenceT3():
    mu_x = 55
    sigma_x = 5
    M = 10000
    N = 1500000
    n = 120
    successCounter = []
    B = np.random.normal(mu_x, sigma_x, N)
    
    for i in range(0, M):
        X = B[random.sample(range(N), n)]
        x_bar = np.mean(X)
        s_hat = np.std(X, ddof=1)
        
        mu_lower = x_bar - 2.62 * (s_hat/math.sqrt(n))
        mu_upper = x_bar + 2.62 * (s_hat/math.sqrt(n))
    
        if(mu_x > mu_lower and mu_x < mu_upper):
            successCounter.append(i)

    successProb = len(successCounter)/M
    print("The percentage of success with 99% confidence using t distribution n =", n, "is", successProb)

ninetyNineConfidenceT3()        