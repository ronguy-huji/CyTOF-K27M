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
import matplotlib.cm as cmx
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
from matplotlib import colors
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
    return np.std(od['H3.3'])+np.std(od['H4'])+np.std(od['H3'])
                  #(pow(od['H3']-avMarkers,2)+pow(od['H3.3']-avMarkers,2)+pow(od['H4']-avMarkers,2))







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
        plt.scatter(u[:,0], u[:,1], c=cc,s=3,cmap=plt.cm.jet)
        plt.clim(-5,5)
        plt.colorbar()
    plt.title(title, fontsize=18)
    return u;

#df=pd.read_csv("control.csv")
#dfmut=pd.read_csv("mutant.csv")
dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets/"
C15=pd.read_csv(dir+"C06.csv")
C16=pd.read_csv(dir+"C08.csv")


#C11=C11[(C11 != 0).all(1)]
#C12=C12[(C12 != 0).all(1)]



NamesAll=['H3',
 'H3K36me3',
 'H2B',
 'H3K4me3',
# 'pHistone H2A.X [Ser139]',
 'H3K36me2',
 'H4K16Ac',
 'H2AK119ub',
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
 'pHistone H3 [S28]']



plt.figure(figsize=(6, 5))

GateColumns=['H3',
 'H3K36me3',
 'H2B',
 'H3K4me3',
# 'pHistone H2A.X [Ser139]',
 'H3K36me2',
 'H4K16Ac',
 'H2AK119ub',
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
 'pHistone H3 [S28]']


            
C15=C15[(C15[['H3','H3.3','H4']]>5).all(axis=1)]
C16=C16[(C16[['H3','H3.3','H4']]>5).all(axis=1)]



C15=C15[(C15[GateColumns]>0).all(axis=1)]
C16=C16[(C16[GateColumns]>0).all(axis=1)]

C15=C15[GateColumns]
C16=C16[GateColumns]



C16_z=(C16 - C16.mean())/C16.std()
C15_z=(C15 - C15.mean())/C15.std()


C16_NO=C16[(C16_z<=5).all(axis=1)]
C15_NO=C15[(C15_z<=5).all(axis=1)]


C16=C16_NO
C15=C15_NO


sns.set_style({'legend.frameon':True})
#CAll=CAll[CAll['clust'].isin([0,1])]
d0=C15.std()/C15.mean()
d1=C16.std()/C16.mean()



dd0=np.log(d0.sort_values())
dd1=np.log(d1.sort_values())

# fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
# ax.hlines(y=dd0.index, xmin=-5, xmax=5, color='gray', alpha=0.7, 
#           linewidth=1, linestyles='dashdot')

# ax.scatter(y=dd0.index, x=dd0, s=50, c='blue', alpha=0.7,
#            label="C15",)
# ax.scatter(y=dd0.index, x=d1, s=50, c='green', alpha=0.7,
#            label="C16",)
# #ax.scatter(y=dd2.index, x=dd2, s=900*np.power(sz1,2), c='red', alpha=0.7,
# #           label="Cluster 2",)
# #ax.scatter(y=dd3.index, x=dd3, s=900*np.power(sz3,2), c='magenta', alpha=0.7,
# #           label="Cluster 3",)



# ax.vlines(x=0, ymin=0, ymax=len(dd0)-1, color='black', alpha=0.7, linewidth=2, linestyles='dotted')
# plt.legend(fontsize=20,
#            facecolor='White', framealpha=1,frameon=True)

# ax.set_title('$\sigma$/$\mu$', fontdict={'size':22})
# ax.set_xlim(0, 12.5)

# plt.show()



diffs=(dd1-dd0).sort_values(ascending=False)    
 
clrs = ['red' if x < 0 else 'green' for x in diffs]

NMS=[
 'H3K36me3',
 'H3K4me3',
# 'pHistone H2A.X [Ser139]',
 'H3K36me2',
 'H4K16Ac',
 'H2AK119ub',
 'H3K4me1',
 'H3K64ac',
 'H3K27ac',
 'cleaved H3',
 'H3K9ac',
# 'H3K27me3',
 'H3K9me3',
# 'pHistone H3 [S28]'
 ]

diffs=diffs[NMS]
diffs=diffs.sort_values(ascending=False)    
clrs = ['red' if x < 0 else 'green' for x in diffs]

# Draw plot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=clrs, alpha=0.4, linewidth=5)

# Decorations
plt.gca().set(ylabel='$Marker$', xlabel='$Difference$')
#plt.yticks(df.index, df.cars, fontsize=12)
plt.title('$log(\sigma/\mu)$ Mutant - WT', fontdict={'size':20})
plt.grid(linestyle='--', alpha=0.5)
plt.show()





dd0=np.log(d0[NMS].sort_values())
dd1=np.log(d1[NMS].sort_values())

jet = cm = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=0, vmax=len(dd0))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
print(scalarMap.get_clim())





fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.hlines(y=[0,1], xmin=0, xmax=6, color='gray', alpha=0.7, 
          linewidth=1, linestyles='dashdot')
for count, value in enumerate(dd0.index):
    print(count)
    colorVal = scalarMap.to_rgba(count)
    ax.scatter(y=1, x=dd0[NMS][value], s=50, color=colorVal, alpha=0.7,label=value)
    ax.scatter(y=0, x=dd1[value], s=50, color=colorVal, alpha=0.7)
    ax.plot([dd0[value],dd1[value]],[1,0],color=colorVal,linewidth=2)

ax.set_xlim(-1.5,0)
plt.yticks([0,1], ['C16','C15'], rotation='horizontal')    
plt.legend()
plt.title('$log(\sigma/\mu)$')

C15=C15.assign(type='WT'); 
C16=C16.assign(type='Mutant');
CAll=C15.append(C16)