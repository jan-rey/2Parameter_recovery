#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Janne Reynders; janne.reynders@ugent.be
2: true alpha and eps are sampled randomly from a uniform distribution (in script with same name, alpha and eps et 10 values linearly spaced between 0 and 10 and then every combination is made)
"""

from scipy.optimize import minimize # finding optimal params in models
from scipy import stats             # statistical tools
import os                           # operating system tools
import numpy as np                  # matrix/array functions
import pandas as pd                 # loading and manipulating data
import matplotlib.pyplot as plt     # plotting
import math

#Frist: simulate a Rescorla Wagner model with constant epsilon and constant learning rate (alpha) in a variable context (the 8 choice context from Jensen and Neuringer)
#to have data for which we will try and recover the parameters alpha and eps
def simulate_RW_variable(alpha, eps, T, Q_int):

    #alpha      --->        learning rate
    #eps        --->        epsilon
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
 
    #the environment presents 8 choices. A reward is given whenever the current choice completes a duplet (a sequence of two sebsequent choices) of which the frequency count is lower than a certain threshold
    #therefore, for each possible two choice sequence, we keep a frequency that gets updated every trial (frequencies are initialized at 20)
    #seq_options represents all possible 64 two choice sequences
    seq_options = np.array([[0,0],
                            [0,1],
                            [0,2],
                            [0,3],
                            [0,4],
                            [0,5],
                            [0,6],
                            [0,7],
                            [1,0],
                            [1,1],
                            [1,2],
                            [1,3],
                            [1,4],
                            [1,5],
                            [1,6],
                            [1,7],
                            [2,0],
                            [2,1],
                            [2,2],
                            [2,3],
                            [2,4],
                            [2,5],
                            [2,6],
                            [2,7],
                            [3,0],
                            [3,1],
                            [3,2],
                            [3,3],
                            [3,4],
                            [3,5],
                            [3,6],
                            [3,7],
                            [4,0],
                            [4,1],
                            [4,2],
                            [4,3],
                            [4,4],
                            [4,5],
                            [4,6],
                            [4,7],
                            [5,0],
                            [5,1],
                            [5,2],
                            [5,3],
                            [5,4],
                            [5,5],
                            [5,6],
                            [5,7],
                            [6,0],
                            [6,1],
                            [6,2],
                            [6,3],
                            [6,4],
                            [6,5],
                            [6,6],
                            [6,7],
                            [8,0],
                            [8,1],
                            [8,2],
                            [8,3],
                            [8,4],
                            [8,5],
                            [8,6],
                            [8,7]])
    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    Freq = np.ones((K_seq), dtype = float)*20
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice   
    
    for t in range(T):
        #store values for Q
        Q_k_stored[t,:] = Q_k   
              
        #make choice based on choice probababilities
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(Q_k)
        if rand[t] == 1:
            k[t] = np.random.choice(range(K))

        #reward schedule based on frequencies of two-choice sequences
        if t < 1:
            r[t] = 1
        else: 
            current_seq = k[t-1:t+1]
            current_index = np.where(np.all(seq_options==current_seq,axis=1))[0]
            Adding = np.ones(64, dtype=float)*(-1/63)
            Freq = np.add(Freq, Adding)
            Freq[current_index] = Freq[current_index] + 1 + (1/63)
            if Freq[current_index] < 21.6:
                r[t] = 1
                Freq = Freq*0.984
            else:
                r[t] = 0
              
        # update values
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + alpha * delta_k    

    return k, r, Q_k_stored



#Second, define the negative log likelihood function
def negll_RW_eps(alpha, k, r, eps):
    Q_int = 1
   
    K = np.max(k)+1
    T = len(k)
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)
    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):
        Q_k_stored[t,:] = Q_k
        max_Q = np.argmax(Q_k)
        if k[t] == max_Q: #this indicates the greedy option was taken, with a probability of 1-epsilon
            choice_prob[t] = 1-eps + (eps/K) #probability to choose greedy option + probability to choose option with highest Q value in a non-greedy state
        else: #this indicates a random choice (non-greedy) option was chosen
            choice_prob[t] = eps/K

        # update Q values
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + alpha * delta_k


    negLL = -np.sum(np.log(choice_prob)) 

    return negLL
    

#Third, simuate data
attempt = 1
T=500
Q_int = 1
eps=0.1
total_sim = 100
results = pd.DataFrame(index=range(0, total_sim), columns=['true_alpha', 'recov_alpha', 'negLL', 'BIC'])

for count in range(total_sim):
    true_alpha = np.random.rand()

    k, r, Q_k_stored = simulate_RW_variable(alpha = true_alpha, eps=eps, T=T, Q_int=Q_int)
    results.at[count, 'true_alpha'] = true_alpha

#Fourth, do parameter recovery

    negLL = np.inf #initialize negative log likelihood

    guess = 0.5

    result = minimize(negll_RW_eps, x0=guess, args=(k,r,eps), bounds=[(0, 1)])
    negLL = result.fun
    param_fits = result.x
    
    BIC = np.log(T) + 2*negLL
   
    print('checkpoint', param_fits)
    #store in dataframe
    results.at[count, 'recov_alpha'] = param_fits[0]
    results.at[count, 'negLL'] = negLL
    results.at[count, 'BIC'] = BIC

    
    print('checkpoint', count)

save_dir = '/Users/jareynde/OneDrive - UGent/1A_Main_Project/Models/1Random/epsilon/2Param_recov/output_constant/1Neuringer_env/8choices/alpha'


title_excel = os.path.join(save_dir, f'{attempt}_results.xlsx')
results.to_excel(title_excel, index=False)
results = pd.read_excel(title_excel)

#columns are 'true_alpha', 'recov_alpha', 'negLL', 'BIC'
results = results.to_numpy()
true_alpha = results[0:, 0]
recov_alpha = results[0:, 1]
negLL = results[0:, 2]
BIC = results[0:, 3]


def calculate_rmse(actual_values, predicted_values):
    n = len(actual_values)
    squared_diffs = [(actual_values[i] - predicted_values[i]) ** 2 for i in range(n)]
    mean_squared_diff = sum(squared_diffs) / n
    rmse = math.sqrt(mean_squared_diff)
    return rmse
def calculate_Rsquared(y_true, y_pred):
    mean_y_true = np.mean(y_true)
    ssr = sum((y_true[i] - y_pred[i])**2 for i in range(len(y_true)))
    sst = sum((y_true[i] - mean_y_true)**2 for i in range(len(y_true)))
    Rsquared = 1 - (ssr / sst)
    return Rsquared


RMSE_LR = calculate_rmse(true_alpha, recov_alpha)
Rsquared_LR = calculate_Rsquared(true_alpha,recov_alpha)
print('RMSE LR is', RMSE_LR)
print('Rsquared LR is', Rsquared_LR)
fig, ax1 = plt.subplots(figsize=(15,8))


ax1.scatter(x=true_alpha, y=recov_alpha)
ax1.set_title('parameter recovery of learning rate in variable context')
ax1.set_xlabel('true learning rate')
ax1.set_ylabel('recovered learning rate')
fig_name1 = os.path.join(save_dir, f'{attempt}_true-vs-recov')
plt.savefig(fig_name1)
plt.show()


