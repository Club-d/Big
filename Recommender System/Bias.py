import pandas as pd

def Dictionary( Data, Col_name ): #Create the dictionary for "id" and paired index.

    content = set()

    for item in Data:
        content.add(item[Col_name])#Add the unique value to the set ().

    columns = list(content) #Convert set to list.
    
    dic = {}
    for index, element in enumerate(columns):
        dic[element] = index #Create a dictionary.

    return dic
    
def Global_Bias( Data ) : #Calculate global bias
    global_score = 0
    training_numbers = 0

    for item in Data: #Read rating data by row
        rating = item['rating']
        global_score += rating
        training_numbers +=1 

    global_bias = global_score / training_numbers
    
    return global_bias

def User_Bias(Data, Usr_Dic): # Calculate the user bias

    df = pd.DataFrame(Data)[['user_id','rating']] #Convert Data to dataframe 

    df['user_id'] = df['user_id'].map(Usr_Dic) #Replace the "user_id" with corresponding index 

    ubias_score = df.groupby('user_id')['rating'].sum() #Group by "user_id " and aggregate the rating

    uid_count = df['user_id'].value_counts() #Return a list by counting "user_id" 

    b_user = ubias_score / uid_count.where(uid_count != 0, 0) - Global_Bias(Data) #User bias Calculation

    return b_user

def Item_Bias(Data, Item_Dic):# Calculate the item bias
    
    df = pd.DataFrame(Data)[['book_id','rating']] #Convert Data to dataframe

    df['book_id'] = df['book_id'].map(Item_Dic) #Replace the "book_id" with corresponding index 

    bbias_score = df.groupby('book_id')['rating'].sum() #Group by "book_id " and aggregate the rating

    bid_count = df['book_id'].value_counts()  #Return a list by counting "book_id" 

    b_item = bbias_score / bid_count.where(bid_count != 0, 0) - Global_Bias(Data) #Item bias Calculation
    
    return b_item