'''
Created 4/10/2018
@author: anon

Assignment 2: Counting Distinct Elements

Citations: Cardinality Estimation from http://blog.notdot.net/2012/09/Dam-Cool-Algorithms-Cardinality-Estimation
'''

import numpy as np
import math
import time


def trailing_zeroes(num):
    """Counts the number of trailing 0 bits in num."""
    if num == 0:
        return 32
    p = 0
    while (num >> p) & 1 == 0:
        p += 1
    return p


def loglog(values, k):
    """Estimates the number of unique elements in the input set values.

    Arguments:
        values: An iterator of hashable elements to estimate the cardinality of.
        k: The number of bits of hash to use as a bucket number; there will be 2**k buckets.
    """
    np.random.seed(17)
    vals=[np.random.randint(0,2**32) for i in range(values)]  #values to be hashed
    num_buckets = 2 ** k
    max_zeroes = [0] * num_buckets
    for value in np.arange(values):
        h = vals[value]
        bucket = h & (num_buckets - 1) # Mask out the k least significant bits as bucket ID
        h_value= h >> k
        #print bin(h_value)
        max_zeroes[bucket] = max(max_zeroes[bucket], trailing_zeroes(h_value))
    #print 'estimated value:',2 ** (float(sum(max_zeroes)) / num_buckets) * num_buckets * 0.79402
    return 2 ** (float(sum(max_zeroes)) / num_buckets) * num_buckets * 0.79402


def flajolet_martin(values):
    #np.random.seed(17)
    num_hash=2**7
    #num_hash gives the number of hash functions to be tested
    R_list=[0]*num_hash  #list of highest number of trailing zeroes
    for h in range(num_hash):
        k=30  #number of bits for each hash value
        hash_vals = np.matrix([[np.random.randint(0, 2) for i in range(k)] for j in range(values)])
        #randomy generated hash values
        R=0
        for value in np.arange(values):
            arr = np.array(hash_vals[value, :])[0] ##makes an array out of each matrix row
            b = ''.join(map(str, arr))  #turns array into string
            h_value = int(b, 2)  #turns string of binary to integer
            R_list[h] = max(R_list[h], trailing_zeroes(h_value))  #keeps track of highest R
    means=[]
    #here i split
    for j in np.arange(0,len(R_list),int(ceil(np.log2((values))))):
        part_list=R_list[j:j+int(ceil(np.log2((values))))]
        means.append(np.mean(part_list))
    #print 'estimated value',2**np.median(means)
    return 2**np.median(means)


if __name__ == '__main__':
    #print ('Relative Approximation Errors for the 3 algorithms.')
    num_unique = np.random.randint(1000,2**17)
    print ('Number of unique values:',num_unique)
    #print ('probabilistic counting:')
    #print ('%.3f' % np.mean([np.abs(1 - (1/0.7751)*probabilistic_counting(num_unique)/num_unique) for j in range(1)]))
    print ('loglog:')
    start_time = time.time()
    print ('%.3f' % ((np.abs(1- np.mean([loglog(num_unique, 10)/num_unique for j in range(1)])))))
    print('Running time %s seconds'  % (time.time() - start_time))
    print ('Flajolet-Martin:')
    start_time = time.time()
    print ('%.3f'% np.abs(1-np.mean([(flajolet_martin(num_unique)/num_unique) for j in range(1)])))
    print('Running time %s seconds'  % (time.time() - start_time))


'''
import random
 
def randomHash(modulus):
   a, b = random.randint(0,modulus-1), random.randint(0,modulus-1)
   def f(x):
      return (a*x + b) % modulus
   return f
 
def average(L):
   return sum(L) / len(L)
 
def numDistinctElements(stream, numParallelHashes=10):
   modulus = 2**20
   hashes = [randomHash(modulus) for _ in range(numParallelHashes)]
   minima = [modulus] * numParallelHashes
   currentEstimate = 0
 
   for i in stream:
      hashValues = [h(i) for h in hashes]
      for i, newValue in enumerate(hashValues):
         if newValue < minima[i]:
            minima[i] = newValue
 
      currentEstimate = modulus / average(minima)
 
      yield currentEstimate


S = [random.randint(1,2**20) for _ in range(10000)]
 
for k in range(10,301,10):
   for est in numDistinctElements(S, k):
      pass
   print(abs(est))


S = range(10000) #[random.randint(1,2**20) for _ in range(10000)]
 
for k in range(10,301,10):
   for est in numDistinctElements(S, k):
      pass
   print(abs(est))
'''
