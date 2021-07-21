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


def dbscan_plot(data,eps=0.1,min_samples=50):
    X=data
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))

    # Black removed and is used for noise instead.
    plt.figure(figsize=(10, 10))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),label = k,
                 markeredgecolor='k', markersize=14)
        
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    
    plt.legend(fontsize=15, title_fontsize='40')    
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    return labels




#df=pd.read_csv("control.csv")
#dfmut=pd.read_csv("mutant.csv")
dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets/17321/"
df1=pd.read_csv(dir+"C16_Norm.csv")
df2=pd.read_csv(dir+"C17_Norm.csv")


NamesAll=['H3',
 'IdU',
 'MBP',
 'H3K36me3',
 'GFAP',
 'EZH2',
 'H3K4me3',
 'H3K79me2',
 'pHistone H2A.X [Ser139]',
 'H3K36me2',
 'Sox2',
 'SIRT1',
 'H4K16ac',
 'H2AK119ub',
 'H3K4me1',
 'H3.3',
 'H3K64ac',
 'BMI1',
 'Cmyc',
 'H4',
 'H3K27ac',
 'H4K20me3',
 'DLL3',
 'cleaved H3',
 'H3K9ac',
 'H1.0',
 'CD24',
 'H3K27me3',
 'H3K27M',
 'H3K9me3',
 'CD44',
 'Ki-67',
 'CXCR4',
 'pHistone H3 [S28]',
 'H1']




plt.figure(figsize=(6, 5))








aaaa=df1.assign(Rep='C16')
aaaa=aaaa.append(df2.assign(Rep='C17'))
df=aaaa

####
Names=['H2AK119ub',
 'H3',
 'H3.3',
 'H3K27M',
 'H3K27me3',
 'H3K36me2',
 'H3K4me1',
 'H3K4me3',
 'H3K64ac',
 'H3K9ac',
 'H3K9me3',
 'H4',
 'H4K16ac',
 'cleaved H3',
 'H3K79me2',
 'H4K20me3',
 'pHistone H3 [S28]']


Names2=['H3',
 'MBP',
 'H3K36me3',
 'GFAP',
 'EZH2',
 'H3K4me3',
 'H3K79me2',
 'pHistone H2A.X [Ser139]',
 'H3K36me2',
 'Sox2',
 'SIRT1',
 'H4K16ac',
 'H2AK119ub',
 'H3K4me1',
 'H3.3',
 'H3K64ac',
 'BMI1',
 'Cmyc',
 'H4',
 'H3K27ac',
 'H4K20me3',
 'DLL3',
 'cleaved H3',
 'H3K9ac',
 'H1.0',
 'CD24',
 'H3K27me3',
 'H3K27M',
 'H3K9me3',
 'CD44',
 'Ki-67',
 'CXCR4',
 'pHistone H3 [S28]']

ddf=df.sample(frac=1)

ar=ddf[Names2]#np.asarray(ddf)
tsne = TSNE(n_components=2, random_state=0,perplexity=40)

#mask = np.random.choice([False, True], len(ar), p=[0.8, 0.2])

X_2d = tsne.fit_transform(ar)#[mask,:])


for NN in NamesAll:
    Var=NN
    TSNEVar=NN
    cc=ddf[TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:,0],X_2d[:,1],s=2,
                c=cc, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(-3.5,3.5)
    plt.title(TSNEVar+" C16/C17 Full Panel tSNE")



X=X_2d
km = KMeans(n_clusters=10,tol=1e-5)
km.fit(X)
km.predict(X)
plt.figure(figsize=(10, 10))
labels = km.labels_#Plotting
u_labels = np.unique(labels)
for i in u_labels:
    plt.scatter(X[labels == i , 0] , X[labels == i , 1] , label = i,s=100)
plt.legend(fontsize=15, title_fontsize='40')
#plot.legend(loc=2)
#plt.setp(ax.get_legend().get_texts(), fontsize='22')

#plt.show()


aaa=ddf.assign(lbl=labels)
aaa=aaa.assign(tsne0=X[:,0])
aaa=aaa.assign(tsne1=X[:,1])


lab=dbscan_plot(X_2d,eps=0.181,min_samples=50)
#aaaa=aaa.assign(clust=lab)

C16=aaa[aaa['Rep']=='C16']
C17=aaa[aaa['Rep']=='C17']


plt.figure(figsize=(6, 5))
plt.scatter(C16['tsne0'],C16['tsne1'],s=2,
            color='red',label="C16")
plt.scatter(C17['tsne0'],C17['tsne1'],s=2,
            color='blue',label="C17")
plt.legend()
plt.title("C16/C17 Full Panel tSNE")