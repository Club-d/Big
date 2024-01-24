

def Misra_Gries(data, size):

    Mis_Gri_dic = {}
    decrement = 0
    
    for element in data:

        if len(Mis_Gri_dic) < size: #if the bucket is not full 
            
            if element in Mis_Gri_dic:
                Mis_Gri_dic[ element ] = Mis_Gri_dic[element] + 1 #if this category exists in the dictionary, then count the true frequency 
                
            else:
                Mis_Gri_dic[ element] = 1 #initialise the new category frequency as 1
           
            continue

        if len(Mis_Gri_dic) == size: #if the bucket is already full
            
            if element in Mis_Gri_dic:
                Mis_Gri_dic[ element ] = Mis_Gri_dic[element] + 1 #if this category exists in the dictionary, then count the true frequency 

            else: 
                decrement = decrement + 1 #count decrement step
                
                Mis_Gri_dic = {key: value - 1 for key, value in Mis_Gri_dic.items()} #existing category deduct 1 frequency 

                delete_keys = [key for key, value in Mis_Gri_dic.items() if value == 0]#if the frequency equals to 0, then this category needs to be deleted 
                
              
                
                for key in delete_keys:
                    del Mis_Gri_dic[key]

            continue
                                    
    return Mis_Gri_dic,decrement