# python
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 00:07:51 2016

@author: ZZK
"""
from numpy import *
import numpy as np
import random
from matplotlib import pylab
import time  
import matplotlib.pyplot as plt  
def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))
def distMat(X,Y):
    mat=[]
    for x in X:
        mat.append(map(lambda y:dist(x,y),Y))
    return np.mat(mat)
def sum_dist(data,label,center):
    s=0
    for i in range(data.shape[0]):
        s+=dist(data[i],center[label[i]])
    return s

def kmeans(data,cluster,threshold=1.0e-19,maxIter=100):
    m=len(data)
    labels=np.zeros(m)
    center=np.array(random.sample(data,cluster))
    s=sum_dist(data,labels,center)
    print s
    #iterator times
    n=0
    print center
    while 1:
        n=n+1
        tmp_mat=distMat(data,center)
        labels=tmp_mat.argmin(axis=1)
        color=['r*','w^','y+']
        pylab.hold(False)
        for i in range(cluster):
            idx=(labels==i).nonzero()
            #print "idx is",idx[0]
            #print data[idx[0]]
            center[i]=np.mean(data[idx[0]],axis=1)
            #center[i]=data[idx[0]].mean(axis=0)
            d_i=data[idx[0]]
            d_i=d_i[0] 
            #print 'd_i',d_i[0:-1,0]
            pylab.plot(d_i[0:-1,0],d_i[0:-1,1],color[i])
            pylab.hold(True)
            print 'center[i] ',center[i][0]
            pylab.scatter(center[i][0],center[i][1],s=1000,marker='.',c='r')
        pylab.show()
        s1=sum_dist(data,labels,center)
        print s1
        if s-s1<=threshold:
            break
        s=s1
        if n>maxIter:
            break
    print n
    return center

## step 1: load data  
print "step 1: load data..."  
dataSet = []  
fileIn = open('H:/data/testSet.txt')  
for line in fileIn.readlines():  
    lineArr = line.strip().split('\t')  
    dataSet.append([float(lineArr[0]), float(lineArr[1])])  
  
## step 2: clustering...  
print "step 2: clustering..."  
dataSet = mat(dataSet)  
k = 4  
center=kmeans(data,k)
print center
