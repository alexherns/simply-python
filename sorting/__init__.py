import math

is_sorted= lambda l: all(l[i] <= l[i+1] for i in xrange(len(l)-1))

def insertion_sort(l):
    """
    Uses insertion sort to sort l
    """
    if not l:
        return None
    for i in range(1, len(l)): 
        for j in range(1, i+1)[::-1]:
            if l[j] < l[j-1]:
                l[j], l[j-1]= l[j-1], l[j]
            else:
                break
    return l

def selection_sort(l):
    """
    Uses selection sort to sort l
    """
    for i in range(len(l)):
        minval= min(l[i:])
        minval= l.pop(i + l[i:].index(minval))
        l.insert(i, minval)
    return l

def merge_sort(l):
    """
    Uses merge sort to sort l
    """
    if len(l) <= 1:
        return l
    k= len(l)/2
    l1, l2= l[:k], l[k:]
    return merged(merge_sort(l1), merge_sort(l2))

def merged(l1, l2):
    """
    Merges two sorted lists
    """
    i, j= 0, 0
    out= []
    n, m= len(l1), len(l2)
    while i < n and j < m:
        if l1[i] <= l2[j]:
            out.append(l1[i])
            i+= 1
        else:
            out.append(l2[j])
            j+= 1
    for li in l1[i:n]: out.append(li)
    for lj in l2[j:m]: out.append(lj)
    return out

def quicksort(l):
    """
    uses quicksort to sort l given lo and hi (this is a wrapper to simplify
    calling)
    """
    return _quick_recursion(l, 0, len(l))

def _quick_recursion(l, lo, hi):
    """
    Recursive program for quicksort
    """
    if lo +1 >= hi:
        return l
    p= partition(l, lo, hi)
    _quick_recursion(l, lo, p)
    _quick_recursion(l, p+1, hi)
    return l

def pivot_func(l):
    """
    An implementation of a pivot function for quick sort, using the triple-median method
    Returns:
        index of pivot
    """
    n= len(l)
    med= sorted([l[0], l[-1], l[n/2]])[1]
    return med
    if med == l[0]:
        return 0
    elif med == l[-1]:
        return n-1
    elif med == l[n/2]:
        return n/2

def partition(l, lo, hi):
    """
    An implementation of a partition function for quick sort
    Returns:
        Two partitions, where items in l1 are <= l[k] and items in l2 are > l[k]
    """
    k= pivot_func(l[lo:hi])
    l1= [i for i in l[lo:hi] if i < k]
    l2= [i for i in l[lo:hi] if i > k]
    l[lo:hi]= l1+[i for i in l[lo:hi] if i == k]+l2
    x= l[lo:hi].index(k)+lo
    return x

def bubble_sort(l):
    """
    Uses bubble sort to sort l, returning a copy of sorted list
    """
    l_copy= list(l)
    switch= False
    while not switch:
        switch= True
        for i in range(len(l_copy)-1):
            if l_copy[i] > l_copy[i+1]:
                switch= False
                l_copy[i], l_copy[i+1]= l_copy[i+1], l_copy[i]
    return l_copy

def shell_sort(l, gaps=[701, 301, 132, 57, 23, 10, 4, 1]):
    """
    Uses shell sort to sort l, returning sorted list
    """
    l= list(l)
    for gap in gaps:
        if gap >= len(l):
            continue
        pieces= []
        for offset in range(gap):
            l[offset::gap]= insertion_sort(l[offset::gap])
    return l

def comb_sort(l, scale=1.3):
    """
    Uses comb sort to sort l, returning a copy of sorted list
    """
    from math import ceil
    l= list(l)
    gap= int(ceil(len(l)/scale))
    while gap > 1:
        for i in range(len(l)-gap):
            if l[i] > l[i+gap]:
                l[i], l[i+gap]= l[i+gap], l[i]
        gap= min(gap-1, int(ceil(gap/scale)))
    l= bubble_sort(l)
    return l

def counting_sort(l):
    """
    Uses counting sort to sort l (a list of integers only!)
    """
    hist= {i:0 for i in range(min(l), max(l)+1)}
    for i in l:
        hist[i]+= 1
    for i in range(min(l)+1, max(l)+1):
        hist[i]+= hist[i-1]
    output= [0 for _ in range(len(l))]
    for i in l[::-1]:
        j= hist[i]
        hist[i]-= 1
        output[j-1]= i
    return output

def bucket_wrapper(l):
    """
    A simple wrapper for bucket sorting l that has an average size of 10 in
    each bucket
    """
    return bucket_sort(l, len(l)/10)

def bucket_sort(l, n=50):
    """
    Bucket sort of l, with n buckets
    """
    M= max(l)+1
    buckets= [[] for _ in range(n)]
    for i in l:
        buckets[bin_this(i, n, M)].append(i)
    print buckets
    output= []
    for i in range(n):
        if not buckets[i]:
            continue
        buckets[i]= insertion_sort(buckets[i])
        output.extend(buckets[i])
    return output

def bin_this(i, n, M):
    """
    Returns bin number of i, if n bins divide M
    """
    return int(i*n/M)

def evaluate_algo(fhandle):
    """
    Performs simple timing metric of the algorithm
    """
    import time
    import numpy as np
    ns= np.arange(100, 10000, 300)
    for n in ns:
        n_times= []
        array= [random.randint(1, n) for _ in range(n)]
        for _ in range(10):
            start= time.time()
            fhandle(array)
            end= time.time()
            n_times.append(end-start)
        print n, np.mean(n_times)

def radix_sort(l):
    """
    Uses radix sort to return a sorted l. Currently only implemented for
    integer lists
    """
    l= list(l)
    i= 0
    while True:
        buckets= radix_distribute(l, i)
        if len(buckets) == 1:
            return buckets[0]
        l= [item for sublist in buckets for item in sublist]
        i+= 1

def radix_distribute(l, i):
    """
    A bucketing method for radix sort of integer lists
    """
    buckets= [[] for _ in range(10)]
    for item in l:
        buckets[(item/10**i)%10].append(item)
    return [bucket for bucket in buckets if bucket]

def heap_sort(l):
    """
    Uses heap sort to return a sorted l
    """
    l= list(l)
    l= heapify(l)
    end= len(l)-1
    while end > 0:
        l[0], l[end]= l[end], l[0]
        end+= -1
        sift_down(l, 0, end)
    return l

def heapify(l):
    """
    Subroutine for heap_sort to generate a max-heap list of l
    """
    start= (len(l)-1)/2
    while start >= 0:
        sift_down(l, start, len(l)-1)
        start+= -1
    return l

def sift_down(l, start, end):
    """
    Subroutine for heap_sort and heapify to move item at start index of l to proper
    place in max-heap of l, terminating before the end index
    """
    root= start
    while root*2 + 1 <= end:
        child= root*2 + 1
        swap= root
        if l[swap] < l[child]:
            swap= child
        if child+1 <= end and l[swap] < l[child+1]:
            swap= child+1
        if swap == root:
            return
        else:
            l[root], l[swap]= l[swap], l[root]
            root= swap


if __name__ == '__main__':
    import random
    #evaluate_algo(heap_sort)
