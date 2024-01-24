import numpy as np
import pandas as pd
import random as rd

from scipy.sparse import csc_matrix

def Power_Interation(Nodes, Edges, Error):

    print('Creating the sparse matrix...\n')
    
    #row denote destinations
    row = [item[1] for item in Edges]

    #column denote sources
    col = [item[0] for item in Edges]
    data = np.ones(len(Edges))

    #Compressed Sparse Column matrix initialisation
    Cscmatrix = csc_matrix((data,(row, col)),shape = (Nodes,Nodes))

    print(f'{Cscmatrix.todense()}')
        
    print('Calculating the summation of the columns...')
    
    column_sums = Cscmatrix.sum(axis=0)
    print(f'{column_sums}')
   
    print('Initialising the weight of the sparse matrix...')
    for col_idx in range(Cscmatrix.shape[1]):
        if column_sums[0,col_idx] !=0:
            Cscmatrix.data[Cscmatrix.indptr[col_idx]:Cscmatrix.indptr[col_idx + 1]] /= column_sums[0,col_idx]

    print('Completed.\n')
    
    print(f'{Cscmatrix.todense()}')
    
    print('Initialising the vector_0...')

    #initialise the vector r
    r = np.ones(Nodes)
    r0 = r /sum(r)
    print('Completed.\n')
    
    r_n =r0
 
    Err = float('inf')
    iterations = 0  
    print('Looping...')
    Vectors = {}
    Vectors[0] = r0

    #iterate r
    while Err >= Error:   
        
        iterations += 1
        if iterations % 50 == 0:
            print(f'This is the iterations of {iterations}.\n')
        r_n1 = Cscmatrix.dot(r_n)
        
        Delta = r_n1 - r_n
        
        #Error in L1 normal form
        Err = np.linalg.norm(Delta, ord=1)
        r_n = r_n1
        
        Vectors[iterations] = r_n
    
    print('The final result has been generated!')
    return Cscmatrix,column_sums, iterations, Vectors

def Power_Interation_with_Teleport(Nodes, Edges, Error, Beta):
    
    print('Creating the sparse matrix...\n')
    #row denote destinations
    row = [item[1] for item in Edges]
    
    #column denote sources
    col = [item[0] for item in Edges]
    data = np.ones(len(Edges))

    #Compressed Sparse Column matrix initialisation
    Cscmatrix = csc_matrix((data,(row, col)),shape = (Nodes,Nodes))

    print(f'{Cscmatrix.todense()}')
        
    print('Calculating the summation of the columns...')

    column_sums = Cscmatrix.sum(axis=0)
    print(f'{column_sums}')
   
    print('Initialising the weight of the sparse matrix...')
    for col_idx in range(Cscmatrix.shape[1]):
        if column_sums[0,col_idx] !=0:
            Cscmatrix.data[Cscmatrix.indptr[col_idx]:Cscmatrix.indptr[col_idx + 1]] /= column_sums[0,col_idx]
    
    print('Completed.\n')

    print('Initialising the vector_0...')

    #initialise the vector r
    r = np.ones(Nodes)
    r0 = r /sum(r)
    print('Completed.\n')
    
    r_old =r0
    r_new = np.zeros(Nodes)
    
    Vectors = {}
    Vectors[0]  =r0
    
    S = 0 
    
    Err = float('inf')
    iterations = 0  
    print('Looping...')
    N = Nodes

    print(f'Beta is {Beta}.\n')
 
    while Err >= Error:   

        iterations += 1
        if iterations % 50 == 0:
            print(f'This is the iterations of {iterations}.\n')

        #Step 1
        r_new = Beta * Cscmatrix.dot(r_old) 

        #Step 2
        S = np.sum(r_new, axis=0)

        #Step 3
        r_new = r_new + (1 - S) / N

        #Error L1 normal form
        Delta = r_new - r_old
        Err = np.linalg.norm(Delta, ord=1) 
        r_old = r_new
        
        Vectors[iterations] = r_new

    print('The final result has been generated!')
    return Cscmatrix,column_sums, iterations, Vectors