import math

import pandas as pd
import random as rd
import numpy as np

def Interaction_dataframe( Data, Row_dic, Col_dic ): # The real rating dataframe, include the "user_id","book_id","rating"

    df = pd.DataFrame(Data)[['user_id','book_id','rating']]
    
    user = df['user_id'].map(Row_dic)
    book = df['book_id'].map(Col_dic)
    rating = df['rating']

    interaction = pd.concat([user, book, rating], axis=1)

    return interaction

def Latent_factor(Length, Factor): #Initialise the latent factor

    latent = np.random.randn(Length,Factor) # Length: the number of documents, Factor : the length of factor

    return latent

def SGD_LFM(R, Q, P, Factor, Iterations, lambda1, lambda2, eta): # Regularise latent factor model
    
    t = 1 
    RMSE = {}
    
    while t <= Iterations: 
        errors = []
        
        for index, row in R.iterrows():

            #Return the corresponding "user_id" and "item_id"
            uid = row['user_id']  
            itid = row['book_id']
                  
            pred = Q[uid].dot(P[itid].T) #Calculate the predicted rating based on latent factor

            RE = row['rating'] - pred #The RE infers the difference between the real rating and the predicted rating 
            errors.append(RE)
            
            for f in range(Factor): 
                
                #Stochastic gradient descent
                grad_qi = -2 * RE * P[ itid, f ] + 2 * lambda1 * Q[ uid, f ]
                grad_pj= -2 * RE * Q[ uid, f ]  + 2 * lambda2 * P[ itid, f ]  

                Q[ uid, f ] = Q[ uid, f ] - eta * grad_qi
                P[ itid, f ] = P[ itid, f ] - eta * grad_pj
          
        RMSE[ t ] = round ((math.sqrt(sum([x ** 2 for x in errors]) / R['rating'].size)),6) #Root means square error calculation
        t += 1             
        
    return RMSE

def SGD_LFM_bias(R, Q, P, Bg, Bi, Bj, Factor, Iterations, lambda1, lambda2, lambda3, lambda4, eta): # Regularise latent factor model with bias terms
    
    t = 1 
    RMSE = {}
    
    while t <= Iterations: 
        errors = []
        
        for index, row in R.iterrows():
            
            #Return the corresponding "user_id" and "item_id"
            uid = row['user_id']
            itid = row['book_id']

            pred = Q[uid].dot(P[itid].T) + Bg + Bi[uid] + Bj[itid]  #Calculate the predicted rating based on latent factor, bg, bi and bj bias terms.

            RE = row['rating'] - pred #The RE infers the difference between the real rating and the predicted rating 
            errors.append(RE)
            
            for f in range(Factor): 
                
                #Stochastic gradient descent for P and Q
                grad_qi = -2 * RE * P[ itid, f ] + 2 * lambda1 * Q[ uid, f ]
                grad_pj = -2 * RE * Q[ uid, f ]  + 2 * lambda2 * P[ itid, f ]  

                Q[ uid, f ] = Q[ uid, f ] - eta * grad_qi
                P[ itid, f ] = P[ itid, f ] - eta * grad_pj

            #Stochastic gradient descent for Bi and Bj
            grad_bi = -2 * RE + 2 * lambda3 * Bi[uid]
            grad_bj = -2 * RE + 2 * lambda4 * Bj[itid]
               
            Bi[uid] = Bi[uid] - eta * grad_bi
            Bj[itid] = Bj[itid] - eta * grad_bj
          
        RMSE[ t ] = round ((math.sqrt(sum([x ** 2 for x in errors]) / R['rating'].size)),6) #Root means square error calculation
        t += 1             
        
    return RMSE, Bi, Bj, P, Q
    
def GD_LFM(R, Q, P, Factor, Iterations, lambda1, lambda2, eta): # Regularise latent factor model
    
    t = 1 
    RMSE = {}
    
    while t <= Iterations: 
        
        errors = []
        
        for index, row in R.iterrows():

            #Return the corresponding "user_id" and "item_id"
            uid = row['user_id']  
            itid = row['book_id']
                  
            pred = Q[uid].dot(P[itid].T) #Calculate the predicted rating based on latent factor

            RE = row['rating'] - pred #The RE infers the difference between the real rating and the predicted rating 
            errors.append(RE)

        E = sum(errors) / R.shape[0]
            
        for f in range(Factor): 
                
            #gradient descent
            grad_qi = -2 * E * P[ itid, f ] + 2 * lambda1 * Q[ uid, f ]
            grad_pj= -2 * E * Q[ uid, f ]  + 2 * lambda2 * P[ itid, f ]  

            Q[ uid, f ] = Q[ uid, f ] - eta * grad_qi
            P[ itid, f ] = P[ itid, f ] - eta * grad_pj
          
        RMSE[ t ] = round ((math.sqrt(sum([x ** 2 for x in errors]) / R['rating'].size)),6) #Root means square error calculation
        t += 1             
        
    return RMSE

def GD_LFM_bias(R, Q, P, Bg, Bi, Bj, Factor, Iterations, lambda1, lambda2, lambda3, lambda4, eta): # Regularise latent factor model with bias terms
    
    t = 1 
    RMSE = {}
    
    while t <= Iterations: 
        errors = []
        
        for index, row in R.iterrows():
            
            #Return the corresponding "user_id" and "item_id"
            uid = row['user_id']
            itid = row['book_id']

            pred = Q[uid].dot(P[itid].T) + Bg + Bi[uid] + Bj[itid]  #Calculate the predicted rating based on latent factor, bg, bi and bj bias terms.

            RE = row['rating'] - pred #The RE infers the difference between the real rating and the predicted rating 
            errors.append(RE)

        E = sum(errors) / R.shape[0]
        
        for f in range(Factor): 
                
                #Gradient descent for P and Q
                grad_qi = -2 * E * P[ itid, f ] + 2 * lambda1 * Q[ uid, f ]
                grad_pj = -2 * E * Q[ uid, f ]  + 2 * lambda2 * P[ itid, f ]  

                Q[ uid, f ] = Q[ uid, f ] - eta * grad_qi
                P[ itid, f ] = P[ itid, f ] - eta * grad_pj

            #Gradient descent for Bi and Bj
            grad_bi = -2 * E + 2 * lambda3 * Bi[uid]
            grad_bj = -2 * E + 2 * lambda4 * Bj[itid]
               
            Bi[uid] = Bi[uid] - eta * grad_bi
            Bj[itid] = Bj[itid] - eta * grad_bj
          
        RMSE[ t ] = round ((math.sqrt(sum([x ** 2 for x in errors]) / R['rating'].size)),6) #Root means square error calculation
        t += 1             
        
    return RMSE, Bi, Bj, P, Q 
