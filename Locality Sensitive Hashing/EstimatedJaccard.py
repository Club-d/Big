import sys

import numpy as np
import random as rd


def Is_Prime(random_number):
    
    if  random_number < 2:
        return False
    for i in range(2, int(random_number ** 0.5) + 1):
        if random_number % i == 0:
            return False;
    return True;
    
def MinHash(k, n):#hash size k, the number of features n 
    
    #Initialise Prime number as a relatively large number 
    while True:
        generate_number = rd.getrandbits(30) #get a random number, then test if it is a prime number
        if Is_Prime(generate_number): # if not, re-generate a new number  
            break

    p = generate_number
    a = []
    b = []
    
    for i in range( k ):
        b_i = rd.randint(0, p - 1)
        a_i = rd.randint(1, p - 1)
        
        a.append(a_i)
        b.append(b_i)
    
    print(f'when k = { k } ,the minHash function we generated are:\n')
    
    for i in range( k ):
        #universal hash function is h(x) = ((a x +b) mod p) mod n
        print(f'(({ a[i] } * x + { b[i] }) mod { p }) mod { n })')
    print("\n")
    
    return a,b,p

def Simulated_Permutation(a,b,p,n,k):  #coeffiecent for min hash function: a,b,p, the number of features n, the hash size k,

    hashValue = np.zeros((k,n))

    #the number of row that we need to calculate simulated permulation on the same minhash function
    for i in range(k):
        
        #the number of column that equals to the number of features  
        for x in range(n):

            # the simulated permutation value on the corresponding position
            hashValue[i][x]= int(((a[i] * x + b[i]) % p )% n)
    
    return hashValue

def Signature_Matrix(data, hashValue, n, D):
    #data: shingles documents, hashValue: simulated permutation, the number of features: n, the number of articles: D 
    
    sigMtrix = []
    sigLen = len(hashValue)

    #signature matrix size equals the number of articles *  k row 
    for i in range(sigLen):
        sigMtrix.append([sys.maxsize] * D)
    
    for i in range(sigLen):  
        for d in range(D):
            for j in range(n):

                #If the shingles are equal to 0, then do nothing
                if data[d][j] == 0:
                    continue

                #If the shingles are equal to 1, always select minimal permutation pi
                else:  
                    if sigMtrix[i][d] > hashValue[i][j]:
                        sigMtrix[i][d] = hashValue[i][j]
    
    return sigMtrix

def Hash_Table(sigmtrix, m): #sigmtrix: signature matrix we generated, the total number of buckets the articles will put in :m

    #Initialise Prime number as a relatively large number 
    while True:
        generate_number = rd.getrandbits(30) #get a random number, then test if it is a prime number
        if Is_Prime(generate_number): # if not, re-generate a new number  
            break

    p = generate_number
    c = []
    G = []
    Table = {}

    # r: the number r dimensions of signature matrix  
    r = len(sigmtrix)
    d = len(sigmtrix[0])
    
    #create the coefficients of g1 function
    for k in range(r + 1) :
        if k  == 0 :
            c.append( rd.randint(0, p - 1))
        else:
            c.append( rd.randint(1, p - 1))

    c0 = c[0]
    c1 = c[1:] 
    
    for i in range(d):
       #c0 + c1 * x1 + c2 * x2 + c3 * x3 +.... +cn *Xn
       g = c0 + np.dot(c1 , sigmtrix[:,i])
       # (sum(g) mod p ) mod m 
       G.append(g % p % m)

    for ind, num in enumerate(G):
        if num not in Table:
            Table[ int(num) ] = [ ind ]
        else:
            Table[ int(num) ]. append(ind)

    return Table    

def HST_Search(g_tbl,q_id): 
    #q_tbl:query bucket number ,g_tbl: global bucket number ,q_id :query id

    #create the collision dictionary
    collision_dic = {}

    #scan the query id in corresponding bucket and report the index
    for index in (q_id - 1):
        for key, value in sorted(g_tbl.items()):
           if index in value:
               collision_dic[ key] = value 
               
    return collision_dic

def Estimated_Jaccard_Similarity(QuerySig, CandiSig): #QuerySig: query signature matrix, CandiSig:candidate signature matrix
    
    ets_score = [] 
    if len(CandiSig) == len(QuerySig):
    
        index = 0
        rows = len(CandiSig)
        for ind in range(len(CandiSig[0])):
            sim = 0
            for row in range(rows):

                #if the signature is matched, the similarity add 1.
                if CandiSig[row][ind] == QuerySig[row]:
                    sim = sim + 1
            #The correponding estimated jaccard similarity is: the total number row of same signature / the row of total scores
            ets_score.append(sim/rows)
            
    return ets_score