'''
@author: anon

Assignment 2: Counting Distinct Elements

Citations: Cardinality Estimation from http://blog.notdot.net/2012/09/Dam-Cool-Algorithms-Cardinality-Estimation
'''

import numpy as np
import random
import math
import time

def relError(size,cardinality):
    return np.abs(size-cardinality)/size

def trailing_zeroes(num):
    """
    Counts the number of trailing 0 bits in hashed value 'num'
    """
    if num == 0:
        return 0 #return 32 or 0?
    p = 0
    while (num >> p) & 1 == 0:
        p += 1
    return p

def LogLog(vals,k):
    """
    estimates the number of unique elements in the input set values
    using the durand-flajolet (2003) algorithm.

    arguments:
        values: an iterator of hashable elements to estimate the cardinality of
        k: the number of bits of hash to use as a bucket number

    parameters:
        m: 2**k buckets
        vals: to find out
        M_zeroes: M-elements of data stream, 1 for each bucket
    """
    alpha = 0.79402
    m = 2**k
    M_zeroes = [0]*m # initialize M^(1),...,M^(m) to 0
    #print(M_zeroes)
    for value in vals:
        h = hash(value)
        bucket = h & (m - 1) # mask out the k least significant bits as bucket ID
        h_value= h >> k
        #print bin(h_value)
        M_zeroes[bucket] = max(M_zeroes[bucket], trailing_zeroes(h_value))
    return(2**(float(sum(M_zeroes))/m)*m*alpha)

def pcsa(values):
    '''
    implementation of the flajolet-martin algorithm, or the
    probabilistic counting with stochastic averaging. algorithm described
    in flajolet & martin (1985).
    '''
    R = trailing_zeroes(values)
    return 2**R
    # m = 2**10 #number of hash functions to be tested
    # phi = 0.77351 #magic number phi
    # R_list = [0]*m  #list of highest number of trailing zeroes
    # for h in range(m):
    #     k = 32  #number of bits for each hash value
    #     hash_vals = np.matrix([[np.random.randint(0, 1) for i in range(k)] for j in range(values)]) #randomly generated hash values
    #     R = 0
    #     for value in np.arange(values):
    #         arr = np.array(hash_vals[value,:])[0] #makes an array out of each matrix row
    #         b = ''.join(map(str,arr))  #turns array into string
    #         h_value = int(b,2)  #turns string of binary to integer
    #         R_list[h] = max(R_list[h], trailing_zeroes(h_value))  #keeps track of highest R
    # S = []
    # #here i split
    # for j in np.arange(0,len(R_list),int(math.ceil(np.log2((values))))):
    #     part_list=R_list[j:j+int(math.ceil(np.log2((values))))]
    #     S.append(np.mean(part_list))
    # return((m/phi)*(2**np.median(S)))

"""
iterating k buckets (between 64 bits ie 2^4 and 1024 bits 2^10)
and over m number of buckets in same range,
over n size.
"""
if __name__ == '__main__':
    # np.random.seed(1)
    vals = [random.getrandbits(32) for i in range(10000)]
    #num_unique = np.random.randint(1,2e4) #
    # print(np.size(num_unique))
    # print('true n:',num_unique)
    #start_time = time.time()
    # pcsa_test=[np.abs(10000-pcsa(vals)/10000) for j in range(1)]
    #print('PCSA: %.3f'% np.abs(1-np.mean([(pcsa(num_unique)/num_unique) for j in range(1)])))
    #print('Run time %s seconds'  % (time.time() - start_time))
    #start_time = time.time()
    test = [np.abs(10000-LogLog(vals,10)/10000) for j in range(1)]
    print(test)
    #print('LogLog: %.3f' % ((np.abs(10000-LogLog(vals,5)/10000) for j in range(1))))
    #print('Run time %s seconds'  % (time.time() - start_time))
