'''
@author: anon

Assignment 2: Counting Distinct Elements

Citations: Cardinality Estimation from http://blog.notdot.net/2012/09/Dam-Cool-Algorithms-Cardinality-Estimation
'''

import numpy as np
import math
import time


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


def LogLog(values, k):
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
    vals = [np.random.randint(0,2**30) for i in range(values)]  #values to be hashed
    m = 2**k
    M_zeroes = [0]*m # initialize M^(1),...,M^(m) to 0
    print(M_zeroes)
    for value in np.arange(values):
        h = vals[value]
        bucket = h & (m - 1) # Mask out the k least significant bits as bucket ID
        h_value= h >> k
        #print bin(h_value)
        M_zeroes[bucket] = max(M_zeroes[bucket], trailing_zeroes(h_value))
    #print('estimated value:',2 ** (float(sum(max_zeroes)) / num_buckets) * num_buckets * 0.79402)
    return(2**(float(sum(M_zeroes))/m)*m*0.79402)


def pcsa(values):
    '''
    implementation of the flajolet-martin algorithm, or the
    probabilistic counting with stochastic averaging. algorithm described
    in flajolet & martin (1985).
    '''
    num_hash=2**7 #number of hash functions to be tested
    R_list=[0]*num_hash  #list of highest number of trailing zeroes
    for h in range(num_hash):
        k=32  #number of bits for each hash value
        hash_vals = np.matrix([[np.random.randint(0, 1) for i in range(k)] for j in range(values)]) #randomy generated hash values
        R=0
        for value in np.arange(values):
            arr = np.array(hash_vals[value, :])[0] ##makes an array out of each matrix row
            b = ''.join(map(str, arr))  #turns array into string
            h_value = int(b, 2)  #turns string of binary to integer
            R_list[h] = max(R_list[h], trailing_zeroes(h_value))  #keeps track of highest R
    means=[]
    #here i split
    for j in np.arange(0,len(R_list),int(math.ceil(np.log2((values))))):
        part_list=R_list[j:j+int(math.ceil(np.log2((values))))]
        means.append(np.mean(part_list))
    return 2**np.median(means)


if __name__ == '__main__':
    np.random.seed(1)
    num_unique = np.random.randint(1,2e4) #
    # print(np.size(num_unique))
    # print('true n:',num_unique)
    start_time = time.time()
    print('PCSA: %.3f'% np.abs(1-np.mean([(pcsa(num_unique)/num_unique) for j in range(1)])))
    print('Run time %s seconds'  % (time.time() - start_time))
    start_time = time.time()
    print('LogLog: %.3f' % ((np.abs(1-np.mean([LogLog(num_unique, 20)/num_unique for j in range(1)])))))
    print('Run time %s seconds'  % (time.time() - start_time))
