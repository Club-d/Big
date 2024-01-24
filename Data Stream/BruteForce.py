def Brute_Force(data):#counting the true frequency
    
    frequency = {}

    for cat in data:
        
        if cat in frequency:
            frequency[cat]= frequency[cat] + 1 #if this category exists in the dictionary, then count the true frequency 
            
        else:
            frequency[cat] = 1 #initialise the new category frequency as 1

    return frequency 