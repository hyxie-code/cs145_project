# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a CS145 project.
"""

import autograd
from autograd.builtins import tuple
import autograd.numpy as np
from scipy.integrate import odeint
import csv

def f(y,t,theta):
    S,I = y
    ds = -theta*S*I
    di = theta*S*I - I

    return np.array([ds,di])

#The gradient descent method is adapted from https://dpananos.github.io/posts/2019/05/blog-post-14/

#Jacobian wrt y
J = autograd.jacobian(f,argnum=0)
#Gradient wrt theta
grad_f_theta = autograd.jacobian(f,argnum=2)

def ODESYS(Y,t,theta):

    #Y will be length 4.
    #Y[0], Y[1] are the ODEs
    #Y[2], Y[3] are the sensitivities

    #ODE
    dy_dt = f(Y[0:2],t,theta)
    #Sensitivities
    grad_y_theta = J(Y[:2],t,theta)@Y[-2::] + grad_f_theta(Y[:2],t,theta)

    return np.concatenate([dy_dt,grad_y_theta])

def Cost(y_obs):
    def cost(Y):
        '''Squared Error Loss'''
        n = y_obs.shape[0]
        err = np.linalg.norm(y_obs - Y, 2, axis = 1)

        return np.sum(err)/n

    return cost

def main():
    active = {}
    confirmed = {}
    death_rate = {}
    with open("train_trendency.csv") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            if row[6] != "":
                if row[1] in active:
                    active[row[1]].append(float(row[6]))
                    confirmed[row[1]].append(float(row[3]))
                    death_rate[row[1]].append(float(row[9]))
                else:
                    active[row[1]] = [float(row[6])]
                    confirmed[row[1]] = [float(row[3])]
                    death_rate[row[1]] = [float(row[9])]
            else:
                death_rate[row[1]].append(float(row[9]))
    
    for state in death_rate:
        death_rate[state] = np.average(death_rate[state])
            
            
    pop = {}
    with open("us_population_2021.csv") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            pop[row[1]] = float(row[2])
                
    t = np.arange(54)
    t2 = np.arange(109)
    
    theta_iter = 1.5

    maxiter = 100
    learning_rate = 1
    
    result = {}
    for key in active:
        y_obs = np.array([pop[key] - np.array(confirmed[key]), active[key]]).transpose()
        Y0 = np.array([pop[key]*0.9, pop[key]*0.1, 0.0, 0.0])
        cost = Cost(y_obs[:,:2])
        grad_C = autograd.grad(cost)
        
        for i in range(maxiter):

            sol = odeint(ODESYS,y0 = Y0, t = t, args = tuple([theta_iter]))

            Y = sol[:,:2]

            theta_iter -= learning_rate*(grad_C(Y)*sol[:,-2:]).sum()
        
        sol = odeint(ODESYS, y0 = Y0, t = t2, args = tuple([theta_iter]))
        result[key] = sol[:,1][-30:] - sol[:,1][53] + confirmed[key][53]
        
    with open("submission.csv", mode = "w+", newline = '') as outfile:
        csvwriter = csv.writer(outfile)
        csvwriter.writerow(["ID", "Confirmed", "Deaths"])
        i = 0
        while i < 1500:
            for state in result:
                con = result[state][i//50]
                csvwriter.writerow([i, con, 0.01*con*death_rate[state]])
                i += 1
    
    
if __name__ == "__main__":
    main()
