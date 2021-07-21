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
import matplotlib
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



dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets/17321/"
C11=pd.read_csv(dir+"C16_Norm_UMAP_2Clust.csv")

C11=C11[C11['clust'].isin([0,1,2,3,4])]

ExpName='C18'

Names=NamesAll=['H3',
 'IdU',
 'MBP',
 'GFAP',
 'EZH2',
 'H3K4me3',
 'H3K79me2',
 'yH2A.X',
 'H3K36me2',
 'Sox2',
 'SIRT1',
 'H4K16ac',
 'H2Aub',
 'H3K4me1',
 'H3.3',
 'H3K64ac',
 'BMI1',
 'Cmyc',
 'H4',
# 'PDGFRa',
 'H4K20me3',
 'DLL3',
 'cleaved H3',
 'H3K9ac',
 'H3K27ac',
 'CD24',
 'H3K27me3',
 'H3K27M',
 'H3K9me3',
 'CD44',
 'Ki-67',
 'CXCR4',
 'pH3[S28]',
 'H1.3/4/5']


EpiGenList=NamesAll=['H3',
 'H3K4me3',
 'H3K79me2',
 'yH2A.X',
 'H3K36me2',
 'H4K16ac',
 'H2Aub',
 'H3K4me1',
 'H3.3',
 'H3K64ac',
 'H4',
 'H4K20me3',
 'cleaved H3',
 'H3K9ac',
 'H3K27ac',
 'H3K27me3',
 'H3K27M',
 'H3K9me3',
 'pH3[S28]',
 'H1.3/4/5']


dfmask=pd.read_csv(dir+"C16_Mask.csv")[Names]
#C11[dfmask]=float("nan")
#C11[dfmask]=float("nan")

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

NClust=2

#dfmask=C11==-100
#C11[dfmask]=float("nan")


aaa=C11[C11['clust'].isin([0,1,2,3,4])]


for NN in Names:
    BoxVar=NN
    plt.figure(figsize=(6, 5))    
    ax = sns.boxplot(x="clust", y=NN, data=aaa,showfliers=False)
    plt.title(NN+" "+ExpName+" Epigen")
    

BCK=C11
C11[dfmask]=float("nan")


for NN in Names:
    Var=NN
    TSNEVar=NN
    cc=C11[TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(C11['umap0'],C11['umap1'],s=2,
                c=cc, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(-3.5,3.5)
    mask=C11[TSNEVar].isna()
    cmap = matplotlib.cm.get_cmap('jet')
    rgba = cmap(-10)
    plt.scatter(C11[mask]['umap0'],C11[mask]['umap1'],s=2,
           color=rgba)
    plt.title(TSNEVar+" "+ExpName+" Epigen")
    
    
cList=['blue','red','green','magenta','pink']
plt.figure(figsize=(10,10))
for i in range(0,5):
    mask=C11['clust']==i    
    plt.scatter(C11[mask]['umap0'],C11[mask]['umap1'],s=2,
                c=cList[i],label="Cluster "+str(i))    
plt.legend(markerscale=5);
plt.title(ExpName+" Clusters Epigen");

C11=BCK
 

#C11[C11['clust'].isin([0,1,3])]=0

### Mean of Dist C17


sns.set_style({'legend.frameon':True})

d0=C11[C11['clust']==0]
d1=C11[C11['clust']==1]
d2=C11[C11['clust']==2]
d3=C11[C11['clust']==3]
dd0=np.mean(d0[Names]).sort_values(ascending=False)
dd1=np.mean(d1[Names]).sort_values()
dd2=np.mean(d2[Names]).sort_values()
dd3=np.mean(d3[Names]).sort_values()
sz0=np.full(len(dd0),0,dtype=np.float64)
sz1=np.full(len(dd1),0,dtype=np.float64)
sz2=np.full(len(dd1),0,dtype=np.float64)
sz3=np.full(len(dd1),0,dtype=np.float64)


for i in range(0,len(dd0)):
    print(str(i)+" "+dd0.index[i])
    q=np.std(d0[dd0.index[i]])*1/1.25
    sz0[i]=q
    print(q)
 
    q=np.std(d1[dd1.index[i]])*1/1.25
    sz1[i]=q
    
    q=np.std(d2[dd2.index[i]])*1/1.25
    sz2[i]=q
    
    q=np.std(d3[dd3.index[i]])*1/1.25
    sz3[i]=q

    
    
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.hlines(y=dd0.index, xmin=-5, xmax=5, color='gray', alpha=0.7, 
          linewidth=1, linestyles='dashdot')

ax.scatter(y=dd0.index, x=dd0, s=900*np.power(sz0,2), c='blue', alpha=0.7,
           label="Cluster 0",)
ax.scatter(y=dd1.index, x=dd1, s=900*np.power(sz1,2), c='green', alpha=0.7,
           label="Cluster 1",)
#ax.scatter(y=dd2.index, x=dd2, s=900*np.power(sz1,2), c='red', alpha=0.7,
#           label="Cluster 2",)
#ax.scatter(y=dd3.index, x=dd3, s=900*np.power(sz3,2), c='magenta', alpha=0.7,
#           label="Cluster 3",)



ax.vlines(x=0, ymin=0, ymax=len(dd0)-1, color='black', alpha=0.7, linewidth=2, linestyles='dotted')
plt.legend(fontsize=20,
           facecolor='White', framealpha=1,frameon=True)

ax.set_title(ExpName+' Mean of Distributions', fontdict={'size':22})
ax.set_xlim(-3.5, 2.5)

plt.show()


'''
Names_w_c=[
 'H3K36me3',
 'H3K4me3',
 'H3K79me2',
 'H3K36me2',
 'H4K16ac',
 'H2AK119ub',
 'H3K4me1',
 'H3K64ac',
 'H3K27ac',
 'H4K20me3',
 'cleaved H3',
 'H3K9ac',
 'H3K27me3',
 'H3K27M',
 'H3K9me3',
 'pHistone H3 [S28]',
 'clust']

g=sns.pairplot(C11[C11['clust']!=-1][Names_w_c],hue='clust',palette="tab10",corner=True,
              diag_kind="kde",diag_kws=dict(fill=True, common_norm=False),)

g.fig.suptitle("C16", y=1.08) # y= some height>1
'''

sns.set_style({'legend.frameon':True})

d0=C11[C11['clust']!=1]
d1=C11[C11['clust']==1]
d2=C11[C11['clust']==2]
d3=C11[C11['clust']==3]
dd0=np.mean(d0[Names]).sort_values(ascending=False)
dd1=np.mean(d1[Names]).sort_values()
dd2=np.mean(d2[Names]).sort_values()
dd3=np.mean(d3[Names]).sort_values()
sz0=np.full(len(dd0),0,dtype=np.float64)
sz1=np.full(len(dd1),0,dtype=np.float64)
sz2=np.full(len(dd1),0,dtype=np.float64)
sz3=np.full(len(dd1),0,dtype=np.float64)


for i in range(0,len(dd0)):
    print(str(i)+" "+dd0.index[i])
    q=np.std(d0[dd0.index[i]])*1/1.25
    sz0[i]=q
    print(q)
 
    q=np.std(d1[dd1.index[i]])*1/1.25
    sz1[i]=q
    
    q=np.std(d2[dd2.index[i]])*1/1.25
    sz2[i]=q
    
    q=np.std(d3[dd3.index[i]])*1/1.25
    sz3[i]=q

    
    
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.hlines(y=dd0.index, xmin=-5, xmax=5, color='gray', alpha=0.7, 
          linewidth=1, linestyles='dashdot')

ax.scatter(y=dd0.index, x=dd0, s=900*np.power(sz0,2), c='blue', alpha=0.7,
           label="Cluster 0",)
ax.scatter(y=dd1.index, x=dd1, s=900*np.power(sz1,2), c='green', alpha=0.7,
           label="Cluster 1",)
#ax.scatter(y=dd2.index, x=dd2, s=900*np.power(sz1,2), c='red', alpha=0.7,
#           label="Cluster 2",)
#ax.scatter(y=dd3.index, x=dd3, s=900*np.power(sz3,2), c='magenta', alpha=0.7,
#           label="Cluster 3",)



ax.vlines(x=0, ymin=0, ymax=len(dd0)-1, color='black', alpha=0.7, linewidth=2, linestyles='dotted')
plt.legend(fontsize=20,
           facecolor='White', framealpha=1,frameon=True)

ax.set_title(ExpName+' Mean of Distributions 27M Low vs high', fontdict={'size':22})
ax.set_xlim(-3.5, 2.5)

plt.show()



#### Diverging Bars
sns.set_style({'legend.frameon':True})

d0=C11[C11['clust']==0]
d1=C11[C11['clust']==1]
d2=C11[C11['clust']==2]
d3=C11[C11['clust']==3]


#d0=d0.drop(columns=['H3','H3.3','H4'])
#d1=d1.drop(columns=['H3','H3.3','H4'])


dd0=np.mean(d0[Names]).sort_values(ascending=False)
dd1=np.mean(d1[Names]).sort_values()

dd0=dd0.drop(labels=['H3','H3.3','H4'])
dd1=dd1.drop(labels=['H3','H3.3','H4'])

dd2=np.mean(d2[Names]).sort_values()
dd3=np.mean(d3[Names]).sort_values()
sz0=np.full(len(dd0),0,dtype=np.float64)
sz1=np.full(len(dd1),0,dtype=np.float64)
sz2=np.full(len(dd1),0,dtype=np.float64)
sz3=np.full(len(dd1),0,dtype=np.float64)




for i in range(0,len(dd0)):
    print(str(i)+" "+dd0.index[i])
    q=np.std(d0[dd0.index[i]])*1/1.25
    sz0[i]=q
    print(q)
 
    q=np.std(d1[dd1.index[i]])*1/1.25
    sz1[i]=q
    
    q=np.std(d2[dd2.index[i]])*1/1.25
    sz2[i]=q
    
    q=np.std(d3[dd3.index[i]])*1/1.25
    sz3[i]=q

diffs=(dd0-dd1).sort_values(ascending=False)    
 
colors = ['red' if x < 0 else 'green' for x in diffs]



# Draw plot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=0.4, linewidth=5)

# Decorations
plt.gca().set(ylabel='$Marker$', xlabel='$Difference$')
#plt.yticks(df.index, df.cars, fontsize=12)
plt.title(ExpName+' Cluster 0 - 1 Diffs Epigenetic', fontdict={'size':20})
plt.grid(linestyle='--', alpha=0.5)
plt.show()





#### Low vs High 

sns.set_style({'legend.frameon':True})

d0=C11[C11['clust']!=1]
d1=C11[C11['clust']==1]
d2=C11[C11['clust']==2]
d3=C11[C11['clust']==3]


#d0=d0.drop(columns=['H3','H3.3','H4'])
#d1=d1.drop(columns=['H3','H3.3','H4'])


dd0=np.mean(d0[Names]).sort_values(ascending=False)
dd1=np.mean(d1[Names]).sort_values()

dd0=dd0.drop(labels=['H3','H3.3','H4'])
dd1=dd1.drop(labels=['H3','H3.3','H4'])

dd2=np.mean(d2[Names]).sort_values()
dd3=np.mean(d3[Names]).sort_values()
sz0=np.full(len(dd0),0,dtype=np.float64)
sz1=np.full(len(dd1),0,dtype=np.float64)
sz2=np.full(len(dd1),0,dtype=np.float64)
sz3=np.full(len(dd1),0,dtype=np.float64)




for i in range(0,len(dd0)):
    print(str(i)+" "+dd0.index[i])
    q=np.std(d0[dd0.index[i]])*1/1.25
    sz0[i]=q
    print(q)
 
    q=np.std(d1[dd1.index[i]])*1/1.25
    sz1[i]=q
    
    q=np.std(d2[dd2.index[i]])*1/1.25
    sz2[i]=q
    
    q=np.std(d3[dd3.index[i]])*1/1.25
    sz3[i]=q

diffs=(dd0-dd1).sort_values(ascending=False)    
 
colors = ['red' if x < 0 else 'green' for x in diffs]



# Draw plot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=0.4, linewidth=5)

# Decorations
plt.gca().set(ylabel='$Marker$', xlabel='$Difference$')
#plt.yticks(df.index, df.cars, fontsize=12)
plt.title(ExpName+' K27M High vs Low Diffs Epigenetic', fontdict={'size':20})
plt.grid(linestyle='--', alpha=0.5)
plt.show()

### Cluster map (correlation - ignoring ungated)
C11[dfmask]=float("nan")

aaa=C11[C11['clust'].isin([0,1,2,3,4])]
plt.figure(figsize=(6, 5))


for i in [0,1]:
    print("Cluster map "+str(i))
    plt.figure(figsize=(10,10))
    matrix=aaa[aaa['clust']==i][EpiGenList].corr()
  #  mask = np.triu(np.ones_like(matrix, dtype=bool))

    g=sns.clustermap(matrix, annot=True,
                cmap=plt.cm.jet,vmin=-.2,vmax=1,linewidths=.1); 
 #   mask = mask[np.argsort(g.dendrogram_row.reordered_ind),:]
 #   mask = mask[:,np.argsort(g.dendrogram_col.reordered_ind)]
 #   g=sns.clustermap(matrix, annot=True,
 #               cmap=plt.cm.jet,vmin=-.2,vmax=1,linewidths=.1,mask=mask);
 #   g.ax_col_dendrogram.set_visible(False)
    plt.xticks(rotation=90); 
    plt.yticks(rotation=0); 
    plt.title(ExpName+' Epigen Cluster '+str(i))
    plt.show()


plt.figure(figsize=(10,10))
matrix=aaa[aaa['clust']==1][EpiGenList].corr()-aaa[aaa['clust']==0][EpiGenList].corr()
sns.clustermap(aaa[aaa['clust']==1][EpiGenList].corr()-aaa[aaa['clust']==0][EpiGenList].corr(),annot=True,
            cmap=plt.cm.jet,vmin=-1,vmax=1,linewidths=.1); 
plt.xticks(rotation=90); 
plt.yticks(rotation=0); 
plt.title(ExpName+' EpiGen Cluster 1 - Cluster 0 Correlations')
plt.show()

    