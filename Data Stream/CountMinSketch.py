import random as rd
import numpy as np

def Is_Prime(random_number):
    
    if  random_number < 2:
        return False
    for i in range(2, int(random_number ** 0.5) + 1):
        if random_number % i == 0:
            return False;
    return True;

def Generate_Prime():
   
    while True:
        generate_number = rd.getrandbits(30) #get a random number, then test if it is a prime number
        if Is_Prime(generate_number): # if not, re-generate a new number  
            break

    return generate_number;
    
def CountMin_Sketch(data, width, dimension):
    
    prime1 = Generate_Prime()
    
    print(f'Generated Prime Number are: {prime1} .\n')
    
    print(f'\n{width} is the bucket size of Count Sketch Approach.\n')
    print(f'\n{dimension} independent hash function(h) will be generated.\n')


    coef_ha = []
    coef_hb = []
    coef_sa = []
    coef_sb = []
    length = len(set(data))
    
    h1 = np.zeros((dimension,length))
    hashtable = np.zeros((dimension,width))
    countmin_sketch_fre = np.zeros((length))

    #generate the hash function(h)
    for d in range(dimension):
        b = rd.randint(0, prime1 - 1)
        a = rd.randint(1, prime1 - 1)
       
        print(f'(({ a } * x + { b }) mod { prime1 }) mod {width}')

        for index in range(length):
            h1[d][index] = ((a * index + b) % prime1) % width   


    print(f'\nWe are generating the hash table...\n')

    
    for datum in data:      
        for d in range(dimension):
            location = h1[ d ][ datum ] #lock the location of datum in the hash table
            hashtable[ d ][ location.astype(int)] += 1 #update the hash value of the certain location 
 
    print('The hash table has been generated successfully.\n')

    #update estimated frequency
    for index in range(length):
        ets_fre = []
        
        for d in range(dimension):
            location_fre = h1[ d ][ index ] #lock the location of datum in the countmin sketch table
          
            ets_fre.append(hashtable[ d ][ location_fre.astype(int) ] ) #the correspondent datum estimated frequency array 
            
        countmin_sketch_fre[ index ] = np.min(ets_fre) #the median of estimated frequency array
    
    print('\nThe estimated frequency for all of the items has been generated successfully.\n\n')
            
    return countmin_sketch_fre
    