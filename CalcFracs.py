#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:19:24 2021

@author: ronguy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:09:58 2021

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
import guidata
import guidata.dataset.datatypes as dt
import guidata.dataset.dataitems as di
from tkinter import *
from tkinter.filedialog import asksaveasfilename
import umap
from lmfit import minimize, Parameters

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import itertools

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)



### Graphics Setup
large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
###



dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets/C13C14/"
df=pd.read_csv(dir+"C14_Mask.csv")

df=df[df['clust'].isin([0,1])]



#df=df.replace({'clust':{0:'WT',1:'Mutant 27M Low',2:'Mutant 27M High'}})

ColNames=list(df.columns)
A=df.groupby('clust').sum().drop(columns=['Unnamed: 0'])

for clust in A.index.values:
    lenC=(df['clust']==clust).sum()
    print("Cluster %s has %i rows" % (clust,lenC))
    for NN in A.columns:
        A.loc[clust,NN]=1-A.loc[clust,NN]/lenC
        
A_T=A.T
A=A.assign(clust=A.index.values)
# NMS=[
#  'H3',
#  'H3K36me3',
#  'H2B',
#  'H3K4me3',
#  'pHistone H2A.X [Ser139]',
#  'H3K36me2',
#  'H4K16Ac',
#  'H2AK119ub',
#  'H3K4me1',
#  'H3.3',
#  'H3K64ac',
#  'H4',
#  'H3K27ac',
#  'cleaved H3',
#  'H3K9ac',
#  'H3K27me3',
#  'H3K27M',
#  'H3K9me3',
#  'pHistone H3 [S28]',
#  'clust']
     

B=A#A[NMS]



B=pd.melt(B, id_vars=['clust'])
ColNames=[
 'H3',
 'H3K36me3',
 'H2B',
 'H3K4me3',
 'yH2A.X',
 'H3K36me2',
 'H4K16ac',
 'H2Aub',
 'H3K4me1',
 'H3.3',
 'H3K64ac',
 'H4',
 'H3K27ac',
 'cleaved H3',
 'H3K9ac',
 'H3K27me3',
 'H3K27M',
 'H3K9me3',
 'pH3[S28]',
 ]
for NN in ColNames:
    mask=B['variable']==NN
    fig=plt.figure(dpi= 380, figsize=(3,5))
    ax = fig.add_axes([0,0,1,1])
    Clust = ['H3K27M \nHigh', 'H3K27M \nLow']
    Vals = B[mask].value
    ax.bar(Clust,Vals,color=['darkorange','dimgray'])
    plt.title("C14 " + NN)
    plt.show()

    # ax=sns.barplot(data=B,x='variable',y='value',hue='clust')
    # plt.title('C15/C16', fontsize=30, color='black', alpha=1)
    # plt.ylabel('Fraction expressing modification',fontsize=40)
    # plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
    # plt.setp(ax.get_legend().get_title(), fontsize='32') # for legend title
    # plt.legend(fontsize=30,
    #            facecolor='White', framealpha=1,frameon=True,
    #            loc='upper left', bbox_to_anchor=(1, 0.5))
    # plt.xticks(rotation=90,fontsize=30)
    # plt.yticks(rotation=90,fontsize=30)
    # plt.xlabel('')