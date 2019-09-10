
import numpy as np

#######################################

def init_matrix(size, init, std=None):

    input_size, output_size = size

    if init == 'zero':
        weights = np.zeros(shape=(input_size, output_size))
        
    elif init == 'sqrt_fan_in':
        sqrt_fan_in = np.sqrt(input_size)
        weights = np.random.uniform(low=-1.0/sqrt_fan_in, high=1.0/sqrt_fan_in, size=(input_size, output_size))

    elif init == 'glorot_uniform':
        limit = np.sqrt(6. / (input_size + output_size))
        weights = np.random.uniform(low=-limit, high=limit, size=(input_size, output_size))

    elif init == 'glorot_normal':
        scale = np.sqrt(2. / (input_size + output_size))
        weights = np.random.normal(loc=0.0, scale=scale, size=(input_size, output_size))

    elif init == 'alexnet':
        weights = np.random.normal(loc=0.0, scale=0.01, size=(input_size, output_size))

    elif init == 'normal':
        scale = std
        weights = np.random.normal(loc=0.0, scale=scale, size=(input_size, output_size))

    else:
        weights = np.random.normal(loc=0.0, scale=1.0, size=(input_size, output_size))
        
    return weights
    
#######################################
    
def init_filters(size, init, std=None):

    fh, fw, fin, fout = size

    if init == 'zero':
        weights = np.zeros(shape=(fh, fw, fin, fout))
        
    elif init == 'sqrt_fan_in':
        assert (False)

    elif init == 'glorot_uniform':
        limit = np.sqrt(6. / (fh*fw*fin + fh*fw*fout))
        weights = np.random.uniform(low=-limit, high=limit, size=(fh, fw, fin, fout))

    elif init == 'glorot_normal':
        scale = np.sqrt(2. / (fh*fw*fin + fh*fw*fout))
        weights = np.random.normal(loc=0.0, scale=scale, size=(fh, fw, fin, fout))

    elif init == 'alexnet':
        weights = np.random.normal(loc=0.0, scale=0.01, size=(fh, fw, fin, fout))

    elif init == 'normal':
        scale = std
        weights = np.random.normal(loc=0.0, scale=scale, size=(fh, fw, fin, fout))

    else:
        assert (False)
        
    return weights
    
#######################################
    
    
    
    
    
    
    
    
    
    
