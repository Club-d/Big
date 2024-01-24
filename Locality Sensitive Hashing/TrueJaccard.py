import numpy as np

def True_Jaccard_Similarity(bitvector, query_idx): #bitvector: shingles documents, query_id: query article index

    tru_score = []
    
    #total articles id can be scanned
    for index in range(len(bitvector)):

        #if the query article id equals the current article's id,then do nothing
        if index == (query_idx ):
            continue
            
        elif index != (query_idx ) :
            case_a = 0
            case_abc = 0

            case_a = sum(np.bitwise_and(bitvector[query_idx],bitvector[index]))
            case_abc = sum(np.bitwise_or(bitvector[query_idx],bitvector[index]))
                            
            tru_score.append(round(case_a/case_abc,4))
       
    return tru_score