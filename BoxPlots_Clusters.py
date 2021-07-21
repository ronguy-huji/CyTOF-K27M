#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:18:45 2021

@author: ronguy
"""
import numpy as np
from sklearn.manifold import TSNE
from scipy import integrate as int
from scipy import stats
import matplotlib.pyplot as plt
import time
import shelve
import guidata
import guidata.dataset.datatypes as dt
import guidata.dataset.dataitems as di
import swat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from bootstrap_stat import datasets as d
from bootstrap_stat import bootstrap_stat as bp
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import datasets
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import statsmodels.stats.weightstats as ws
from sklearn.cluster import KMeans




Clus1=[3,6,0,2,8]
Clus2=[4,1,9,7]
Clus3=[5]


Names=['H2AK119ub',
 'H2B',
 'H3',
 'H3.3',
 'H3K27ac',
 'H3K27M',
 'H3K27me3',
 'H3K36me2',
 'H3K36me3',
 'H3K4me1',
 'H3K4me3',
 'H3K64ac',
 'H3K9ac',
 'H3K9me3',
 'H4',
 'H4K16Ac',
 'cleaved H3',
 #'H2A',
 'pHistone H3 [S28]']
lab=np.loadtxt("foo.csv")
dfmut=pd.read_csv("aaa.csv")
df_ = pd.DataFrame(columns=list(dfmut.columns))



for i in Clus1:
    mask=(lab==i)
    ddd=dfmut[mask].assign(clus=1)
    df_=df_.append(ddd)
    
    
for i in Clus2:
    mask=(lab==i)
    ddd=dfmut[mask].assign(clus=2)
    df_=df_.append(ddd)
    
for i in Clus3:
    mask=(lab==i)
    ddd=dfmut[mask].assign(clus=3)
    df_=df_.append(ddd)
    
for NN in Names:
    BoxVar=NN
    plt.figure(figsize=(6, 5))    
    ax = sns.boxplot(x="clus", y=NN, data=df_)
    plt.title(NN)
      
for NN in Names:
    BoxVar=NN
    plt.figure(figsize=(6, 5))    
    ax = sns.violinplot(x="clus", y=NN, data=df_)
    plt.title(NN)
    
#OutName="C14.csv"
#df_.to_csv("C14_Clustered.csv")