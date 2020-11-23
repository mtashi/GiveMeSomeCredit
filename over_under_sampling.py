import pandas as pd
import numpy as np
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def over_under_sampling (Xtrain,ytrain,Xtest,ytest):

    #print(Counter(ytrain))
    #print(Counter(ytest))
    # train oversampling
    over = RandomOverSampler(sampling_strategy=0.6)
    # fit and apply the transform
    Xtrain_over, ytrain_over = over.fit_resample(Xtrain, ytrain)
    # define undersampling strategy
    under = RandomUnderSampler(sampling_strategy='majority')
    # fit and apply the transform
    Xtrain, ytrain = under.fit_resample(Xtrain_over,ytrain_over)
    #print(Counter(ytrain))
    
    
    # test oversampling
    over = RandomOverSampler(sampling_strategy=0.6)
    # fit and apply the transform
    Xtest_over, ytest_over = over.fit_resample(Xtest, ytest)
    # define undersampling strategy
    under = RandomUnderSampler(sampling_strategy='majority')
    # fit and apply the transform
    Xtest, ytest = under.fit_resample(Xtest_over,ytest_over)
    # summarize class distribution
    #print(Counter(ytest))
    
    return Xtrain , ytrain , Xtest, ytest
    

