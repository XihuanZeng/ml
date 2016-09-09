# implement binary Gaussian Naive Bayes classifier

import numpy as np
import random
import math
import sys
from utils import Gaussian_Density

class naive_bayes(object):
    def __init__(self):
        pass

    def fit(self, train_set, train_labels, num_labels):
        self.__num_features = train_set.shape[1]
        for i in range(len(num_labels)):
            subset_index = train_labels







def NB(trainset,testset):

    N1=trainset.shape[0]
    K=trainset.shape[1]
    N2=testset.shape[0]

    ## Gaussian density
    def density(x,mu,sigma):
        if sigma==0:
            return 0
        else:
            const=1/(sigma*(math.sqrt(2*math.pi)))
            exp=((x-mu)**2)/(2*sigma**2)
            result=const*math.exp(-exp)
            return result


 ## estimate the mean and variances using MLE

    OneIndex=np.array(np.where(trainset[:,0]==1)[0])
    ZeroIndex=np.array(np.where(trainset[:,0]==0)[0])

    OnePart=trainset[OneIndex,:]
    ZeroPart=trainset[ZeroIndex,:]

    mu1=OnePart[:,1:].sum(0)/OnePart.shape[0]
    std1=OnePart[:,1:].std(0)
    mu0=ZeroPart[:,1:].sum(0)/ZeroPart.shape[0]
    std0=ZeroPart[:,1:].std(0)

    pi1=float(OneIndex.shape[0])/N1
    pi0=float(1-pi1)

def judgeX(x):

    # This is P(x|Class 1)
    num1=1
    for i in range(K-1):
        a=density(x[i],mu1[i],std1[i])
        if a>0:
            num1*=a

    num0=1
    for i in range(K-1):
        a=density(x[i],mu0[i],std0[i])
        if a>0:
            num0*=a

    P1=num1*pi1
    P0=num0*pi0

    if P1>P0:
        return 1
    else:
        return 0

    result=[]
    for row in testset[:,1:K]:
        value=judgeX(row)
        result.append(value)

    result=np.array(result)
    reality=testset[:,0]

    err=np.count_nonzero(result-reality)
    err_rate=float(err)/N2
    return(err_rate)


def main():
    filename=sys.argv[1]
    num_splits=float(sys.argv[2])/100
    train_percent=np.array(sys.argv[3].split(),dtype='int64')
    file=open(filename)
    data=[]

    try:
        for line in file:
            fields=line.split(',')
            row_data=[float(x) for x in fields]
            data.append(row_data)

        data=np.array(data)
        file.close()
    except Exception as e:
        print('error reading the file')


    N=data.shape[0]
    K=data.shape[1]
    # count number of 0 and 1 in the data
    NumOfOne=np.count_nonzero(data[:,0])
    NumOfZero=N-NumOfOne
    OneIndex=np.array(np.where(data[:,0]==1)[0])
    ZeroIndex=np.array(np.where(data[:,0]==0)[0])
    # train the regression model
    err_rate=[]

    for i in train_percent:
        proportion=float(i)/100
        err=0.0
        for j in range(100):
            # this is the index of 80% training set
            trainIdx1=np.array(random.sample(OneIndex,int(num_splits*NumOfOne)))
            trainIdx2=np.array(random.sample(ZeroIndex,int(num_splits*NumOfZero)))
            trainIdx=np.concatenate([trainIdx1,trainIdx2])

            # this is the index pf 20% test set
            testSet=np.delete(data,trainIdx,axis=0)

            # this is the index of i% among train set, this is our final tranning set
            Index=np.array((random.sample(trainIdx,int(proportion*num_splits*N))))
            # this is our final training set
            trainSet=data[Index,:]
            err=err+NB(trainSet,testSet)

        err_rate.append(float(err)/100)
        print 'for training percent ',i,'%','test err rate is:',float(err)/100

    print(err_rate)

if __name__ == "__main__":
    main()