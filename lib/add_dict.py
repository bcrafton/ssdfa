
import numpy as np
import copy 

def add_dict(dict1, dict2):

    # ret = copy.copy(dict1)

    for key in dict2.keys():
        if key in dict1.keys():
            dict1[key] = dict1[key] + dict2[key]
        else:
            dict1[key] = dict2[key]
            
    return dict1
