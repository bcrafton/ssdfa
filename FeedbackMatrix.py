
import numpy as np
import math

def matrix_rank(mat):
    return np.linalg.matrix_rank(B)
    
def matrix_sparsity(mat):
    # we want to make sure all the [10]s have some sparisty.
    # so we transpose the matrix from [10, 100] -> [100, 10] so that way we can look at the first [10]
    # summing along the 0 axis, sums along the 10s ... so we get a [100] vector sum 
    
    a = np.sum(mat.T[0] != 0)
    assert (np.all(np.sum(mat != 0, axis=0) == a))
    return a
    
def full_feedback(mat):
    # sum along the opposite axis here.
    
    return (np.all(np.sum(mat != 0, axis=1) > 0))

def FeedbackMatrix(size : tuple, sparse : int, rank : int):
    input_size, output_size = size

    if sparse:
        sqrt_fan_out = np.sqrt(1.0 * output_size / np.sqrt(input_size * sparse))
        # sqrt_fan_out = np.sqrt(1.0 * output_size / input_size * sparse)
        # sqrt_fan_out = np.sqrt(output_size)
    else:
        sqrt_fan_out = np.sqrt(output_size)

    high = 1.0 / sqrt_fan_out
    low = -high

    fb = np.zeros(shape=size)
    fb = np.transpose(fb)

    choices = range(input_size)
    counts = np.zeros(input_size)
    total_connects = (1.0 * sparse * rank)
    connects_per = (1.0 * sparse * rank / input_size)
    
    idxs = []
    
    if sparse and rank:
        assert(sparse * rank >= input_size)
        
        # pick rank sets of sparse indexes 
        for ii in range(rank):
            remaining_connects = total_connects - np.sum(counts)
            pdf = (connects_per - counts) / remaining_connects
            pdf = np.clip(pdf, 1e-6, 1.0)
            pdf = pdf / np.sum(pdf)
            
            choice = np.random.choice(choices, sparse, replace=False, p=pdf)
            counts[choice] += 1.
            idxs.append(choice)

        # create our masks
        masks = []
        for ii in range(rank):
            masks.append(np.zeros(shape=(output_size, input_size)))

        for ii in range(output_size):
            choice = np.random.choice(range(len(idxs)))
            idx = idxs[choice]
            masks[choice][ii][idx] = 1.
        
        # multiply mask by random rank 1 matrix.
        for ii in range(rank):
            tmp1 = np.random.uniform(low, high, size=(output_size, 1))
            tmp2 = np.random.uniform(low, high, size=(1, input_size))
            fb = fb + masks[ii] * np.dot(tmp1, tmp2)
            
        # rank fix
        fb = fb * (high / np.std(fb))
        fb = fb.T
        
    elif sparse:
        mask = np.zeros(shape=(output_size, input_size))
        for ii in range(output_size):
            idx = np.random.choice(choices, size=sparse, replace=False)
            mask[ii][idx] = 1.0
        

        mask = mask.T
        fb = np.random.uniform(low, high, size=(input_size, output_size))
        fb = fb * mask

        '''
        mask = mask.T
        fb = np.random.uniform(0.5 / sqrt_fan_out, 2. / sqrt_fan_out, size=(input_size, output_size))
        fb = fb * mask

        sign = np.random.choice([-1., 1.], size=np.prod(np.shape(fb)), replace=True)
        sign = np.reshape(sign, newshape=np.shape(fb))
        fb = fb * sign
        '''

    elif rank:
        fb = np.zeros(shape=(input_size, output_size))
        for ii in range(rank):
            tmp1 = np.random.uniform(low, high, size=(input_size, 1))
            tmp2 = np.random.uniform(low, high, size=(1, output_size))
            fb = fb + np.dot(tmp1, tmp2)
        # rank fix
        fb = fb * (high / np.std(fb))

    else:
        fb = np.random.uniform(low, high, size=(input_size, output_size))

    return fb

'''
size = (10, 100)
rank = 4
sparse = 3

B = FeedbackMatrix(size, sparse, rank)

for rank in range(10):
    for sparse in range(10):
        if rank * sparse > 10:
            B = FeedbackMatrix(size, sparse, rank)
            passed = (sparse == matrix_sparsity(B)) and (rank == matrix_rank(B)) and full_feedback(B)
            print (rank, sparse, passed)
            
            # if not passed:
            #    print (np.sum(B != 0, axis=0))

for rank in range(10):
    B = FeedbackMatrix(size, 0, rank)
    passed = (10 == matrix_sparsity(B)) and (rank == matrix_rank(B)) and full_feedback(B)
    print (rank, 0, passed)

for sparse in range(10):
    B = FeedbackMatrix(size, sparse, 0)
    passed = (sparse == matrix_sparsity(B)) and (10 == matrix_rank(B)) and full_feedback(B)
    print (0, sparse, passed)
'''









