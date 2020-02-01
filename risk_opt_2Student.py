# -*- coding: utf-8 -*-
"""

@author: Steve Xia
"""
import numpy as np
import pandas as pd
from scipy import optimize
#-------------------------------------------------

def te_opt(W_Bench, C, obj_te, c_, b_):
    # function that minimize the objective function
    n = len(W_Bench)
    # change the initial guess to help test whether we find the global optimal
    guess = 2
    #W = rand_weights(n) # start with random weights
    if guess==1:
        W = rand_weights(n) # start with random weights
    elif guess==2:
        W = W_Bench # Use Bench weight as initial guess
    else:
        W = 1/n*np.ones([n,1])
    
    optimized = optimize.minimize(obj_te, W, (W_Bench, C), 
                method='SLSQP', constraints=c_, bounds=b_,  
                options={'ftol':1e-10, 'maxiter': 1000000, 'disp': False})

#    optimized = optimize.minimize(obj_te, W, (W_Bench, C), 
#                method='Nelder-Mead', constraints=c_, bounds=b_,  
#                options={'fatol ':1e-10, 'maxiter': 100000, 'disp': False}) 
#    optimized = optimize.minimize(obj_te, W, (W_Bench, C), 
#                method='Powell', constraints=c_, bounds=b_,  
#                options={'ftol':1e-10, 'maxiter': 100000, 'disp': False})  
#    optimized = slsqp_mine.fmin_slsqp(obj_te, W, args=(W_Bench, C), 
#                eqcons=c_[0], ieqcons=c_[1], bounds=b_,  
#                iter=100000, acc=1.0e-6, iprint=1)
#    minimizer_kwargs = {"method": "BFGS"}
#    ret = basinhopping(func, x0, minimizer_kwargs=minimizer_kwargs,  niter=200)        
        
    if not optimized.success: 
        raise BaseException(optimized.message)
    return optimized.x  # Return optimized weights

def te_opt_n(W_Bench, C, obj_te, c_, b_):
    n = len(W_Bench)
    # change the initial guess to help test whether we find the global optimal
    guess = 2
    # W = rand_weights(n) # start with random weights
    if guess == 1:
        W = rand_weights(n)  # start with random weights
    elif guess == 2:
        W = W_Bench  # Use Bench weight as initial guess
    else:
        W = 1 / n * np.ones([n, 1])

    optimized = optimize.minimize(obj_te, W, (W_Bench, C),
                                  method='SLSQP', constraints=c_, bounds=b_,
                                  options={'ftol': 1e-10, 'maxiter': 1000000, 'disp': False})
    if not optimized.success:
        raise BaseException(optimized.message)
    return optimized.x  # Return optimized weights

def opt_min_te(W, C, b_, c_):
    return(te_opt(W, C, obj_te, c_, b_))

def opt_min_te_n(n, wts, cov):
    return(te_opt_n(n,W, C, obj_te, c_, b_))

    b = constrain_creater(n)
    TElist = []
    for i in b:
        b1_ = i
        # b1_[num_topwtstock_2include:-1] = (0.0,0.0)
        c1_ = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})  # Sum of active weights = 100%
        # Calling the optimizer
        wts_min_trackingerror2 = opt_min_te(wts, cov, b1_, c1_)
        # calc TE achieved
        wts_active2 = wts_min_trackingerror2 - wts
        TE_optimized2 = tracking_error(wts_active2, cov)
        TElist.append(TE_optimized2)
    Min_n_TE = min(TElist)
    return Min_n_TE * 10000
    
def obj_te(W, W_Bench, C): 
    wts_active = W - W_Bench
    return(np.sqrt(np.transpose(wts_active)@C@wts_active))


def port_var(W,C): 
     return(np.dot(np.dot(W, C), W))

def port_vol(W,C): 
     return(np.sqrt(port_var(W,C)))
     
def port_ret(W,R): 
    return(np.dot(R,W))
    
    
def obj_var(W, R, C): 
     return(np.dot(np.dot(W, C), W))

# New SX add   
def obj_varminus(W, R, C): 
     return(-np.dot(np.dot(W, C), W))    
    
def obj_ret(W, R, C): 
     return(-port_ret(W,R))
     

#-------------------------------------------------
# EWMA cov 
def ewma_cov(rets, lamda): 
    T, n = rets.shape
    ret_mat = rets.as_matrix()
    EWMA = np.zeros((T+1,n,n))# corection changed from T to T+1
    S = np.cov(ret_mat.T)  
    EWMA[0,:] = S
    for i in range(1, T+1) :# corection changed from T to T+1
        S = lamda * S  + (1-lamda) * np.matmul(ret_mat[i-1,:].reshape((-1,1)), 
                      ret_mat[i-1,:].reshape((1,-1)))
        EWMA[i,:] = S

    return(EWMA)
    

# create random weights 
def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

