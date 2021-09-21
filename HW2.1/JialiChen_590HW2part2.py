#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 12:49:14 2021

@author: jialichen
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json

#USER PARAMETERS
IPLOT=True
INPUT_FILE='LectureCodes/weight.json'
FILE_TYPE="json"
DATA_KEYS=['x','is_adult','y']
OPT_ALGO='BFGS'

#UNCOMMENT FOR VARIOUS MODEL CHOICES (ONE AT A TIME)
model_type = "logistic"; NFIT=4; xcol=1; ycol=2;

#READ FILE
with open(INPUT_FILE) as f:
    my_input = json.load(f)  #read into dictionary

#CONVERT INPUT INTO ONE LARGE MATRIX (SIMILAR TO PANDAS DF)
X=[];
for key in my_input.keys():
    if(key in DATA_KEYS): X.append(my_input[key])

#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
X=np.transpose(np.array(X))

#SELECT COLUMNS FOR TRAINING 
x=X[:,xcol];  y=X[:,ycol]

#COMPUTE BEFORE PARTITION AND SAVE FOR LATER
XMEAN=np.mean(x); XSTD=np.std(x)
YMEAN=np.mean(y); YSTD=np.std(y)

#NORMALIZE
x=(x-XMEAN)/XSTD;  y=(y-YMEAN)/YSTD; 

#PARTITION
f_train=0.8; f_val=0.2
rand_indices = np.random.permutation(x.shape[0])
CUT1=int(f_train*x.shape[0]); 
train_idx,  val_idx = rand_indices[:CUT1], rand_indices[CUT1:]
xt=x[train_idx]; yt=y[train_idx]; xv=x[val_idx];   yv=y[val_idx]

#MODEL
def model(x,p):
    return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.01))))

#SAVE HISTORY FOR PLOTTING AT THE END
iteration=0; iteration_train=[];iteration_val=[]; loss_train=[];  loss_val=[]

def predict(p):
    global YPRED_T,YPRED_V,MSE_T, MSE_V
    YPRED_T=model(x[train_idx],p)
    YPRED_V=model(x[val_idx],p)
    MSE_T=np.mean((YPRED_T-y[train_idx])**2.0) 
    MSE_V=np.mean((YPRED_V-y[val_idx])**2.0)


#LOSS FUNCTION
#train_use_idx = tidx_use
#val_use_idx = vidx_use
def loss(p, use_idx, list_for_plot,iteration_list):
    global iterations,iteration

    #LOSS Function - train
    yp=model(x[use_idx],p) #model predictions for given parameterization p
    loss=(np.mean((yp-y[use_idx])**2.0)) 
  
    list_for_plot.append(loss)
    iteration_list.append(iteration); iteration+=1

    return loss


def GD_algo(xi,LR,df_dx):
    xip1= xi - LR * df_dx
    return xip1



#INITIAL GUESS
xi=np.random.uniform(0.1,1.,size=NFIT)

#SAVE HISTORY FOR PLOTTING AT THE END
epoch=1; epochs=[]; loss_train=[];  loss_val=[]
def minimizer(f, xi, algo='GD', LR=0.01,method = 'batch'):
    global epoch
    
    #set up initial parameters
    iteration = 1
    dx = 0.0001
    NDIM = len(xi)
    momentum = 0.03
    train_1 =[]
    train_2 =[]
    valid_1 =[]
    valid_2 = []
    
#OPTIMIZATION LOOP
    while (iteration <= 50000):
        if(method == 'batch'):
            if(iteration == 1):
                print('The Optimization process has started.')
            if(iteration >1): 
                epoch += 1
#NUMERICALLY COMPUTE GRADIENT             
            df_dx = np.zeros(NDIM)
            
            for i in range(0,NDIM):
                
                dX = np.zeros(NDIM)
                dX[i] = dx
                xm1 = xi - dX
                xp1 = xi + dX
                
                grad_i = (f(xp1,train_idx,loss_train,iteration_train)-f(xm1,train_idx,loss_train,iteration_train))/dx/2
                
                #record validation loss for plotting
                f(xp1,val_idx,loss_val,iteration_val)
                f(xm1,val_idx,loss_val,iteration_val)
                
                df_dx[i] = grad_i
            
#TAKE AN OPTIMIZER STEP            
            if (algo == 'GD'):
                xip1 = GD_algo(xi,LR,df_dx)
            if (algo == 'MOM'):
                xip1= GD_algo(xi,LR,df_dx) + dx * momentum 
            if(iteration%1==0):
                predict(xi)                    
            xi = xip1
            iteration += 1
                                   
        if(method == 'stochastic'):
            for i in range(0,NDIM):
                dX = np.zeros(NDIM)
                dX[i] = dx
                xm1 = xi - dX
                xp1= xi + dX
                
            for j in range(0,len(val_idx)):
                f(xp1,val_idx[j],loss_val,iteration_val)
                f(xm1,val_idx[j],loss_val,iteration_val)
    
                
            df_dx = np.zeros(NDIM)
#TAKE AN OPTIMIZER STEP
            if (algo == 'GD'):
                xip1 = GD_algo(xi,LR,df_dx)
            if (algo == 'GD+MOM'):
                xip1= GD_algo(xi,LR,df_dx) + dx * momentum  
            if(iteration%1==0):
                predict(xi)
            
            for j in range(0,len(train_idx)):              
                df_dx = np.zeros(NDIM)
                for i in range(0,NDIM):
                    
                    dX = np.zeros(NDIM)
                    dX[i] = dx
                    xm1 = xi - dX
                    xp1=xi + dX
                
                    grad_i = (f(xp1,train_idx[j],loss_train,iteration_train)-f(xm1,train_idx[j],loss_train,iteration_train))/dx/2
                    df_dx[i] = grad_i
            
#TAKE A OPTIMIZER STEP
                if (algo == 'GD'):
                    xip1 = GD_algo(xi,LR,df_dx)
                if (algo == 'MOM'):
                    xip1= GD_algo(xi,LR,df_dx) + dx * momentum   
                if(iteration % 1==0):
                    predict(xi)
            xi = xip1
            iteration += 1            
        if(method == 'mini_batch'):
            for i in range(0,len(train_idx)):
                if i %2 == 0:
                    train_1.append(train_idx[i])
                else:
                    train_2.append(train_idx[i])
            for i in range(0,len(val_idx)):
                if i %2 == 0:
                    valid_1.append(val_idx[i])
                else:
                    valid_2.append(val_idx[i])
    
            if(iteration == 1):
                first_tidx_use = train_1
                second_tidx_use = train_2
                first_vidx_use = valid_1
                second_vidx_use = valid_2 
                print('The optimization process has started')
            if(iteration >1): 
                epoch += 1
                print(epoch)
            
            df_dx = np.zeros(NDIM)
            for i in range(0,NDIM):
                dX = np.zeros(NDIM)
                dX[i] = dx
                xm1 = xi - dX
                xp1=xi + dX
                
                grad_i = (f(xp1,first_tidx_use,loss_train,iteration_train)-f(xm1,first_tidx_use,loss_train,iteration_train))/dx/2
                df_dx[i] = grad_i
                
                f(xp1,first_vidx_use,loss_val,iteration_val)
                f(xm1,first_vidx_use,loss_val,iteration_val)
                
            
#TAKE A OPTIMIZER STEP
            if (algo == 'GD'):
                xip1 = GD_algo(xi,LR,df_dx)
            if (algo == 'MOM'):
                xip1 = GD_algo(xi,LR,df_dx) + dx * momentum  
            if(iteration%1==0):
                predict(xi)                               
            xi = xip1        
            for i in range(0,NDIM):
                dX = np.zeros(NDIM)
                dX[i] = dx
                xm1 = xi - dX
                xp1=xi + dX
                
                grad_i = (f(xp1,second_tidx_use,loss_train,iteration_train)-f(xm1,second_tidx_use,loss_train,iteration_train))/dx/2
                df_dx[i] = grad_i
                
                f(xp1,second_vidx_use,loss_val,iteration_val)
                f(xm1,second_vidx_use,loss_val,iteration_val)
            
#TAKE A OPTIMIZER STEP
            if (algo == 'GD'):
                xip1 = GD_algo(xi,LR,df_dx)
            if (algo == 'MOM'):
                xip1 = GD_algo(xi,LR,df_dx) + dx * momentum 
            if(iteration%1==0):
                predict(xi)
                
            xi = xip1
            iteration += 1
            
            
      
            
    return xi
        
popt = minimizer(loss, xi)  
                        
print("OPTIMAL PARAM:",popt)

#PREDICTIONS
xm=np.array(sorted(xt))
yp=np.array(model(xm,popt))

#UN-NORMALIZE
def unnorm_x(x): 
    return XSTD*x+XMEAN  
def unnorm_y(y): 
    return YSTD*y+YMEAN 

#FUNCTION PLOTS
if(IPLOT):
    fig, ax = plt.subplots()
    ax.plot(unnorm_x(xt), unnorm_y(yt), 'o', label='Training set')
    ax.plot(unnorm_x(xv), unnorm_y(yv), 'x', label='Validation set')
    ax.plot(unnorm_x(xm),unnorm_y(yp), '-', label='Model')
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.legend()
    plt.show()

#PARITY PLOTS
if(IPLOT):
    fig, ax = plt.subplots()
    ax.plot(model(xt,popt), yt, 'o', label='Training set')
    ax.plot(model(xv,popt), yv, 'o', label='Validation set')
    plt.xlabel('y predicted', fontsize=18)
    plt.ylabel('y data', fontsize=18)
    plt.legend()
    plt.show()

#MONITOR TRAINING AND VALIDATION LOSS  
if(IPLOT):
    fig, ax = plt.subplots()
    ax.plot(iteration_train, loss_train, 'o', label='Training loss')
    ax.plot(iteration_val, loss_val, 'o', label='Validation loss')
    plt.xlabel('optimizer iterations', fontsize=18)
    plt.ylabel('loss', fontsize=18)
    plt.legend()
    plt.show()
    