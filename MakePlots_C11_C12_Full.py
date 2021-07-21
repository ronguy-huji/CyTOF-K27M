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



dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets/21321/"
C11=pd.read_csv(dir+"C11_Norm.csv")
C12=pd.read_csv(dir+"C12_Norm.csv")



Names=['MBP', 'PDGFRA', 'DLL3', 'CD24', 'H3K27M', 'CD44', 'CXCR4']




NClust=2

aaa=C11[C11['clust'].isin([0,1,2,3,4])]
plt.figure(figsize=(6, 5))



for NN in Names:
    BoxVar=NN
    plt.figure(figsize=(6, 5))    
    ax = sns.boxplot(x="clust", y=NN, data=aaa,showfliers=False)
    plt.title(NN+" C11")
    



for i in [0,1]:
    plt.figure(figsize=(10,10))
    sns.clustermap(aaa[aaa['clust']==i][Names].corr(),annot=True,
                cmap=plt.cm.jet,vmin=-.2,vmax=1,linewidths=.1); 
    plt.xticks(rotation=90); 
    plt.yticks(rotation=0); 
    plt.title('C11 Cluster '+str(i))
    plt.show()


plt.figure(figsize=(10,10))
sns.clustermap(aaa[aaa['clust']==1][Names].corr()-aaa[aaa['clust']==0][Names].corr(),annot=True,
            cmap=plt.cm.jet,vmin=-1,vmax=1,linewidths=.1); 
plt.xticks(rotation=90); 
plt.yticks(rotation=0); 
plt.title('C11 Cluster 1 - Cluster 0 Correlations')
plt.show()
    
for NN in Names:
    Var=NN
    TSNEVar=NN
    cc=C11[TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(C11['tsne0'],C11['tsne1'],s=2,
                c=cc, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(-3.5,3.5)
    plt.title(TSNEVar+" C11")
    
    
cList=['blue','red']
plt.figure(figsize=(10,10))
for i in range(0,2):
    mask=C11['clust']==i    
    plt.scatter(C11[mask]['tsne0'],C11[mask]['tsne1'],s=2,
                c=cList[i],label="Cluster "+str(i))    
plt.legend(markerscale=5);
plt.title("C11 Clusters");


    
aaa=C12[C12['clust'].isin([0,1,2,3,4])]
plt.figure(figsize=(6, 5))



for NN in Names:
    BoxVar=NN
    plt.figure(figsize=(6, 5))    
    ax = sns.boxplot(x="clust", y=NN, data=aaa,showfliers=False)
    plt.title(NN+" C12")    




for i in [0,1]:
    plt.figure(figsize=(10,10))
    sns.clustermap(aaa[aaa['clust']==i][Names].corr(),annot=True,
                cmap=plt.cm.jet,vmin=-.2,vmax=1,linewidths=.1); 
    plt.xticks(rotation=90); 
    plt.yticks(rotation=0); 
    plt.title('C12 Cluster '+str(i))
    plt.show()

plt.figure(figsize=(10,10))
sns.clustermap(aaa[aaa['clust']==1][Names].corr()-aaa[aaa['clust']==0][Names].corr(),annot=True,
            cmap=plt.cm.jet,vmin=-1,vmax=1,linewidths=.1); 
plt.xticks(rotation=90); 
plt.yticks(rotation=0); 
plt.title('C12 Cluster 1 - Cluster 0 Correlations')
plt.show()

for NN in Names:
    Var=NN
    TSNEVar=NN
    cc=C12[TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(C12['tsne0'],C12['tsne1'],s=2,
                c=cc, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(-3.5,3.5)
    plt.title(TSNEVar+" C12")


cList=['blue','red']
plt.figure(figsize=(10,10))
for i in range(0,2):
    mask=C12['clust']==i    
    plt.scatter(C12[mask]['tsne0'],C12[mask]['tsne1'],s=2,
                c=cList[i],label="Cluster "+str(i))    
plt.legend(markerscale=5);
plt.title("C12 Clusters");



### Mean of Dist C11


sns.set_style({'legend.frameon':True})

d0=C11[C11['clust']==0]
d1=C11[C11['clust']==1]
dd0=np.mean(d0[Names]).sort_values(ascending=False)
dd1=np.mean(d1[Names]).sort_values()
sz0=np.full(len(dd0),0,dtype=np.float64)
sz1=np.full(len(dd1),0,dtype=np.float64)


for i in range(0,len(dd0)):
    print(str(i)+" "+dd0.index[i])
    q=np.std(d0[dd0.index[i]])*1/1.25
    sz0[i]=q
    print(q)
    q=np.std(d1[dd1.index[i]])*1/1.25
    sz1[i]=q

    
    
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.hlines(y=dd0.index, xmin=-5, xmax=5, color='gray', alpha=0.7, 
          linewidth=1, linestyles='dashdot')

ax.scatter(y=dd0.index, x=dd0, s=900*np.power(sz0,2), c='blue', alpha=0.7,
           label="Cluster 0",)
ax.scatter(y=dd1.index, x=dd1, s=900*np.power(sz1,2), c='green', alpha=0.7,
           label="Cluster 1",)



ax.vlines(x=0, ymin=0, ymax=len(dd0)-1, color='black', alpha=0.7, linewidth=2, linestyles='dotted')
plt.legend(fontsize=20,
           facecolor='White', framealpha=1,frameon=True)

ax.set_title('C11 Mean of Distributions', fontdict={'size':22})
ax.set_xlim(-3., 2.5)

plt.show()


### Mean of Dist C12


sns.set_style({'legend.frameon':True})

d0=C12[C12['clust']==0]
d1=C12[C12['clust']==1]
dd0=np.mean(d0[Names]).sort_values(ascending=False)
dd1=np.mean(d1[Names]).sort_values()
sz0=np.full(len(dd0),0,dtype=np.float64)
sz1=np.full(len(dd1),0,dtype=np.float64)


for i in range(0,len(dd0)):
    print(str(i)+" "+dd0.index[i])
    q=np.std(d0[dd0.index[i]])*1/1.25
    sz0[i]=q
    print(q)
    q=np.std(d1[dd1.index[i]])*1/1.25
    sz1[i]=q

    
    
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.hlines(y=dd0.index, xmin=-5, xmax=5, color='gray', alpha=0.7, 
          linewidth=1, linestyles='dashdot')

ax.scatter(y=dd0.index, x=dd0, s=900*np.power(sz0,2), c='blue', alpha=0.7,
           label="Cluster 0",)
ax.scatter(y=dd1.index, x=dd1, s=900*np.power(sz1,2), c='green', alpha=0.7,
           label="Cluster 1",)



ax.vlines(x=0, ymin=0, ymax=len(dd0)-1, color='black', alpha=0.7, linewidth=2, linestyles='dotted')
plt.legend(fontsize=20,
           facecolor='White', framealpha=1,frameon=True)

ax.set_title('C12 Mean of Distributions', fontdict={'size':22})
ax.set_xlim(-3., 2.5)

plt.show()

NMS=['MBP', 'PDGFRA', 'DLL3', 'CD24', 'H3K27M', 'CD44', 'CXCR4','clust']
g=sns.pairplot(C11[C11['clust']!=-1][NMS],hue='clust',palette="tab10",corner=True,
              diag_kind="kde",diag_kws=dict(fill=True, common_norm=False),)

g.fig.suptitle("C11", y=1.08) # y= some height>1


g=sns.pairplot(C12[C12['clust']!=-1][NMS],hue='clust',palette="tab10",corner=True,
               diag_kind="kde",diag_kws=dict(fill=True, common_norm=False),)
g.fig.suptitle("C12", y=1.08) # y= some height>1