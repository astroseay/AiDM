
# coding: utf-8

# In[2]:


'''Implements a straight Jenkins lookup hash - http://burtleburtle.net/bob/hash/doobs.html

Usage:
    from jhash import jhash
    print jhash('My hovercraft is full of eels')

Returns: unsigned 32 bit integer value

Prereqs: None

Tested with Python 2.6.
Version 1.00 - der@dod.no - 23.08.2010

Partly based on the Perl module Digest::JHash
http://search.cpan.org/~shlomif/Digest-JHash-0.06/lib/Digest/JHash.pm

Original copyright notice:
    By Bob Jenkins, 1996.  bob_jenkins@burtleburtle.net.  You may use this
    code any way you wish, private, educational, or commercial.  It's free.

    See http://burtleburtle.net/bob/hash/evahash.html
    Use for hash table lookup, or anything where one collision in 2^^32 is
    acceptable.  Do NOT use for cryptographic purposes.
'''

def mix(a, b, c):
    '''mix() -- mix 3 32-bit values reversibly.
For every delta with one or two bits set, and the deltas of all three
  high bits or all three low bits, whether the original value of a,b,c
  is almost all zero or is uniformly distributed,
* If mix() is run forward or backward, at least 32 bits in a,b,c
  have at least 1/4 probability of changing.
* If mix() is run forward, every bit of c will change between 1/3 and
  2/3 of the time.  (Well, 22/100 and 78/100 for some 2-bit deltas.)'''
    # Need to constrain U32 to only 32 bits using the & 0xffffffff
    # since Python has no native notion of integers limited to 32 bit
    # http://docs.python.org/library/stdtypes.html#numeric-types-int-float-long-complex
    a &= 0xffffffff; b &= 0xffffffff; c &= 0xffffffff
    a -= b; a -= c; a ^= (c>>13); a &= 0xffffffff
    b -= c; b -= a; b ^= (a<<8); b &= 0xffffffff
    c -= a; c -= b; c ^= (b>>13); c &= 0xffffffff
    a -= b; a -= c; a ^= (c>>12); a &= 0xffffffff
    b -= c; b -= a; b ^= (a<<16); b &= 0xffffffff
    c -= a; c -= b; c ^= (b>>5); c &= 0xffffffff
    a -= b; a -= c; a ^= (c>>3); a &= 0xffffffff
    b -= c; b -= a; b ^= (a<<10); b &= 0xffffffff
    c -= a; c -= b; c ^= (b>>15); c &= 0xffffffff
    return a, b, c

def jhash(data, initval = 0):
    '''hash() -- hash a variable-length key into a 32-bit value
  data    : the key (the unaligned variable-length array of bytes)
  initval : can be any 4-byte value, defaults to 0
Returns a 32-bit value.  Every bit of the key affects every bit of
the return value.  Every 1-bit and 2-bit delta achieves avalanche.'''
    length = lenpos = len(data)

    # empty string returns 0
    if length == 0:
        return 0

    # Set up the internal state
    a = b = 0x9e3779b9 # the golden ratio; an arbitrary value
    c = initval        # the previous hash value
    p = 0              # string offset

    # ------------------------- handle most of the key in 12 byte chunks
    while lenpos >= 12:
        a += (ord(data[p+0]) + (ord(data[p+1])<<8) + (ord(data[p+2])<<16) + (ord(data[p+3])<<24))
        b += (ord(data[p+4]) + (ord(data[p+5])<<8) + (ord(data[p+6])<<16) + (ord(data[p+7])<<24))
        c += (ord(data[p+8]) + (ord(data[p+9])<<8) + (ord(data[p+10])<<16) + (ord(data[p+11])<<24))
        a, b, c = mix(a, b, c)
        p += 12
        lenpos -= 12

    # ------------------------- handle the last 11 bytes
    c += length
    if lenpos >= 11: c += ord(data[p+10])<<24
    if lenpos >= 10: c += ord(data[p+9])<<16
    if lenpos >= 9:  c += ord(data[p+8])<<8
    # the first byte of c is reserved for the length
    if lenpos >= 8:  b += ord(data[p+7])<<24
    if lenpos >= 7:  b += ord(data[p+6])<<16
    if lenpos >= 6:  b += ord(data[p+5])<<8
    if lenpos >= 5:  b += ord(data[p+4])
    if lenpos >= 4:  a += ord(data[p+3])<<24
    if lenpos >= 3:  a += ord(data[p+2])<<16
    if lenpos >= 2:  a += ord(data[p+1])<<8
    if lenpos >= 1:  a += ord(data[p+0])
    a, b, c = mix(a, b, c)

    # ------------------------- report the result
    return c

if __name__ == "__main__":
    hashstr = 'My hovercraft is full of eels'
    myhash = jhash(hashstr)
    print ('jhash("%s"): %d' % (hashstr, myhash))

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
