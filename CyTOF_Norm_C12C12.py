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



def residual(params, x, data):
    alpha = params['alpha']
    beta = params['beta']
    gam = params['gamma']
 

    avMarkers=x['H3.3']*alpha+x['H4']*beta+x['H3']*gam
    od=x.subtract(avMarkers,axis=0)
    return np.std(od['H3'])+np.std(od['H3.3'])+np.std(od['H4'])
                  #(pow(od['H3']-avMarkers,2)+pow(od['H3.3']-avMarkers,2)+pow(od['H4']-avMarkers,2))





class RunData(dt.DataSet):
    """
    Execution data
    """
    #
    _bgCool = dt.BeginGroup('Dataset')
    case  = di.StringItem("Case name", default='C14')
    _egCool = dt.EndGroup('Dataset')
    #
#    _bgClu1 = dt.BeginGroup("Cluster 1")
#    u  = di.FloatItem("Velocity [m/s]", min=0, default=20.0)
#    u1  = di.StringItem("Labels 1", default='H2-18O v1')
#    _egClu1 = dt.BeginGroup("Cluster 1")
    _bgJet = dt.BeginGroup("Clusters")
    u1  = di.StringItem("Cluster 1", default='')
    u2  = di.StringItem("Cluster 2", default='')
    u3  = di.StringItem("Cluster 3", default='')
    _egJet = dt.EndGroup("Clusters")



def twoSampZ(X1, X2):
    from numpy import sqrt, abs, round
    from scipy.stats import norm
    mudiff=np.mean(X1)-np.mean(X2)
    sd1=np.std(X1)
    sd2=np.std(X2)
    n1=len(X1)
    n2=len(X2)
    pooledSE = sqrt(sd1**2/n1 + sd2**2/n2)
    z = ((X1 - X2) - mudiff)/pooledSE
    pval = 2*(1 - norm.cdf(abs(z)))
    return round(pval, 4)

def statistic(dframe):
    return dframe.corr().loc[Var1,Var2]


def draw_umap(data,n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''
              ,cc=0):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data);
    plt.figure(figsize=(6, 5))
    if n_components == 2:
        plt.scatter(u[:,0], u[:,1], c=cc,s=3,cmap=plt.cm.seismic)
        plt.clim(-5,5)
        plt.colorbar()
    plt.title(title, fontsize=18)
    return u;

#df=pd.read_csv("control.csv")
#dfmut=pd.read_csv("mutant.csv")
dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets/21321/"
C11=pd.read_csv(dir+"C11.csv")
C12=pd.read_csv(dir+"C12.csv")

#C11=C11[(C11 != 0).all(1)]
#C12=C12[(C12 != 0).all(1)]


Names=['MBP', 'PDGFRA', 'DLL3', 'CD24', 'H3K27M', 'CD44', 'CXCR4']




plt.figure(figsize=(6, 5))









Var1='H3K27M'
Var2='H3K36me2'


scFac=5
C11=np.arcsinh(C11/scFac)
C12=np.arcsinh(C12/scFac)



aaaa=C11.append(C12)
m=np.mean(aaaa)
s=np.std(aaaa)


C11=(C11-m)/s
C12=(C12-m)/s

CAll=C11.assign(Type="C11")
CAll=CAll.append(C12.assign(Type="C12"))

### tSNE C11
'''
ar=C11[Names]#np.asarray(ddf)
tsne = TSNE(n_components=2, random_state=0,perplexity=60)


X_2d = tsne.fit_transform(ar)#[mask,:])

#X_2d=draw_umap(C11[Names],cc=C11['H3K27M'])

for NN in Names:
    Var=NN
    TSNEVar=NN
    cc=C11[TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:,0],X_2d[:,1],s=2,
                c=cc, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(-3.5,3.5)
    plt.title(TSNEVar+" C11")



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

C11=C11.assign(tsne0=X[:,0])
C11=C11.assign(tsne1=X[:,1])
lab=dbscan_plot(X_2d,eps=0.15,min_samples=50)

#C11=C11.assign(clust=lab)
'''


### tSNE C12

ar=C12[Names]#np.asarray(ddf)
tsne = TSNE(n_components=2, random_state=0,perplexity=60,verbose=True)



X_2d = tsne.fit_transform(ar)#[mask,:])

#X_2d=draw_umap(C12[Names],cc=C12['H3K27M'])

for NN in Names:
    Var=NN
    TSNEVar=NN
    cc=C12[TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:,0],X_2d[:,1],s=2,
                c=cc, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(-3.5,3.5)
    plt.title(TSNEVar+" C12")



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

C12=C12.assign(tsne0=X[:,0])
C12=C12.assign(tsne1=X[:,1])

lab=dbscan_plot(X_2d,eps=0.15,min_samples=50)
#C12=C12.assign(clust=lab)
