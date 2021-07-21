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



#df=pd.read_csv("control.csv")
#dfmut=pd.read_csv("mutant.csv")
dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets/17321/"

F1="C18"
F2="C18"



df=pd.read_csv(dir+F1+"_Norm.csv")
dfmut=pd.read_csv(dir+F2+"_Norm.csv")

dfmut_bak=dfmut

Names=list(df.columns)
BlackList = ['lbl','tsne0','tsne1','Unnamed: 0','clust','IdU']

NamesMarkers=[ele for ele in Names if ele not in BlackList] 

'''
['H2AK119ub',
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
 'pHistone H2A.X [Ser139]',
 'pHistone H3 [S28]']
'''



ddf=df[NamesMarkers]


### tSNE Plots
for NN in NamesMarkers:
    Var=NN
    TSNEVar=NN
    cc=ddf[TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(dfmut['tsne0'],dfmut['tsne1'],s=2,
                c=cc, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(-3.5,3.5)
    plt.title("C18 "+TSNEVar)
    plt.savefig("Plots/C18_tSNE_"+NN+".png",dpi=100,bbox_inches = 'tight')


plt.figure(figsize=(6, 5))
plt.scatter(dfmut['tsne0'],dfmut['tsne1'],s=2,
                c=dfmut['clust'], cmap=plt.cm.jet)
plt.clim(-3.5,3.5)
plt.title("Clusters")
plt.colorbar()
plt.savefig("Plots/C18_tSNE_Clust.png",dpi=100,bbox_inches = 'tight')



'''
Names_All=['H2AK119ub',
 'H3K27ac',
 'H3K27me3',
 'H3K36me2',
 'H3K36me3',
 'H3K4me1',
 'H3K4me3',
 'H3K64ac',
 'H3K9ac',
 'H3K9me3',
 'H4K16Ac',
 'cleaved H3',
 'pHistone H2A.X [Ser139]',
 #'H2A',
 'pHistone H3 [S28]']
Open_Chromatin=[
     'H3K4me1', 'H3K4me3', 'H4K16Ac', 'H3K9ac', 'H3K64ac', 'H3K27ac'
     ]
Closed_Chromatin=['H3K9me3', 'H3K27me3', 'H3K36me2', 'H2AK119ub', 'H3K9ac','H3K27ac']

## Cluster 0 Corr Maps
plt.figure(figsize=(10, 10))
g=sns.clustermap(df[Names_All].corr(),
                 annot=True,
                 cmap=plt.cm.jet,
                 vmin=0,vmax=1,linewidths=.1,
                 cbar_pos=(1, 0.5, 0.05, 0.18));
plt.xticks(rotation=90); 
plt.yticks(rotation=0)
g.ax_col_dendrogram.set_visible(False)
#g.ax_col_dendrogram.set_xlim([0,0])
#g.ax_cbar.set_position((0.8, .2, .03, .4))
plt.title(F2+" Cluster 0 All"); 
fig = plt.gcf()  # or by other means, like plt.subplots
figsize = fig.get_size_inches()
fig.set_size_inches(figsize * 1.5)  # scale current size by 1.5
plt.savefig("Plots/"+F2+"Cluster0_All.png",dpi=100,bbox_inches = 'tight')
plt.show()


plt.figure(figsize=(10, 10))
g=sns.clustermap(df[Open_Chromatin].corr(),
                 annot=True,
                 cmap=plt.cm.jet,
                 vmin=0,vmax=1,linewidths=.1,
                 cbar_pos=(1, 0.5, 0.05, 0.18));
plt.xticks(rotation=90); 
plt.yticks(rotation=0)
g.ax_col_dendrogram.set_visible(False)
#g.ax_col_dendrogram.set_xlim([0,0])
#g.ax_cbar.set_position((0.8, .2, .03, .4))
plt.title(F2+" Cluster 0 Open Chromatin");
plt.savefig("Plots/"+F2+"Cluster0_Open.png",dpi=100,bbox_inches = 'tight') 
plt.show()


plt.figure(figsize=(10, 10))
g=sns.clustermap(df[Closed_Chromatin].corr(),
                 annot=True,
                 cmap=plt.cm.jet,
                 vmin=0,vmax=1,linewidths=.1,
                 cbar_pos=(1, 0.5, 0.05, 0.18));
plt.xticks(rotation=90); 
plt.yticks(rotation=0)
g.ax_col_dendrogram.set_visible(False)
#g.ax_col_dendrogram.set_xlim([0,0])
#g.ax_cbar.set_position((0.8, .2, .03, .4))
plt.title(F2+" Cluster 0 Closed Chromatin"); 
plt.savefig("Plots/"+F2+"Cluster0_Closed.png",dpi=100,bbox_inches = 'tight')
plt.show()

## Cluster 1 Corr Maps
plt.figure(figsize=(10, 10))
g=sns.clustermap(dfmut[Names_All].corr(),
                 annot=True,
                 cmap=plt.cm.jet,
                 vmin=0,vmax=1,linewidths=.1,
                 cbar_pos=(1, 0.5, 0.05, 0.18));
plt.xticks(rotation=90); 
plt.yticks(rotation=0)
g.ax_col_dendrogram.set_visible(False)
#g.ax_col_dendrogram.set_xlim([0,0])
#g.ax_cbar.set_position((0.8, .2, .03, .4))
plt.title(F2+" Cluster 1 All"); 
fig = plt.gcf()  # or by other means, like plt.subplots
figsize = fig.get_size_inches()
fig.set_size_inches(figsize * 1.5)  # scale current size by 1.5
plt.savefig("Plots/"+F2+"Cluster1_All.png",dpi=100,bbox_inches = 'tight')
plt.show()


plt.figure(figsize=(10, 10))
g=sns.clustermap(dfmut[Open_Chromatin].corr(),
                 annot=True,
                 cmap=plt.cm.jet,
                 vmin=0,vmax=1,linewidths=.1,
                 cbar_pos=(1, 0.5, 0.05, 0.18));
plt.xticks(rotation=90); 
plt.yticks(rotation=0)
g.ax_col_dendrogram.set_visible(False)
#g.ax_col_dendrogram.set_xlim([0,0])
#g.ax_cbar.set_position((0.8, .2, .03, .4))
plt.title(F2+" Cluster 1 Open Chromatin");
plt.savefig("Plots/"+F2+"Cluster1_Open.png",dpi=100,bbox_inches = 'tight') 
plt.show()


plt.figure(figsize=(10, 10))
g=sns.clustermap(dfmut[Closed_Chromatin].corr(),
                 annot=True,
                 cmap=plt.cm.jet,
                 vmin=0,vmax=1,linewidths=.1,
                 cbar_pos=(1, 0.5, 0.05, 0.18));
plt.xticks(rotation=90); 
plt.yticks(rotation=0)
g.ax_col_dendrogram.set_visible(False)
#g.ax_col_dendrogram.set_xlim([0,0])
#g.ax_cbar.set_position((0.8, .2, .03, .4))
plt.title(F2+" Cluster 1 Closed Chromatin"); 
plt.savefig("Plots/"+F2+"Cluster1_Closed.png",dpi=100,bbox_inches = 'tight')
plt.show()
'''


'''
##Cluster Histograms


df=dfmut_bak[dfmut_bak['clust'].isin([0])]
dfmut=dfmut_bak[dfmut_bak['clust'].isin([1])]
aaa=df.assign(Type='Cluster 0')
aaa=aaa.append(dfmut.assign(Type='Cluster 1'))




size=min(len(dfmut),len(df))
EqSize=aaa[aaa['Type']=="Cluster 0"].iloc[0:size]
EqSize=EqSize.append(aaa[aaa['Type']=="Cluster 1"].iloc[0:size])

for NN in Names:
    sns.kdeplot(data=EqSize,x=NN,hue='Type',fill=True,palette=['red','blue'])
    plt.title(F2+" Clusters "+NN)
    plt.savefig("Plots/"+F2+"_"+NN+"_Clusters_Hist.png",dpi=100,bbox_inches = 'tight')
    plt.show()    

'''


### Means Plot



sns.set_style({'legend.frameon':True})

df0=np.mean(df[df['clust'].isin([0,1])][NamesMarkers]).sort_values()
s0=np.full((len(df0),4),0,dtype=np.float64)
sz0=np.full(len(df0),0,dtype=np.float64)

df1=np.mean(df[df['clust']==2][NamesMarkers]).sort_values()
s1=np.full((len(df1),4),0,dtype=np.float64)
sz1=np.full(len(df1),0,dtype=np.float64)

df2=np.mean(df[df['clust']==2][NamesMarkers]).sort_values()
s2=np.full((len(df2),4),0,dtype=np.float64)
sz2=np.full(len(df2),0,dtype=np.float64)

df3=np.mean(df[df['clust']==3][NamesMarkers]).sort_values()
s3=np.full((len(df3),4),0,dtype=np.float64)
sz3=np.full(len(df3),0,dtype=np.float64)


Srt=df1
Srt=Srt.sort_values(ascending=False)

dbak=df
#df=df[NamesMarkers]

for i in range(0,len(df0)):
    print(str(i)+" "+df0.index[i])
    q=np.std(df[df['clust'].isin([0,1])][df0.index[i]])*1/1.25
    print(q)
    s0[i]=[float(q),0,0,q]
    sz0[i]=q

    print(str(i)+" "+df1.index[i])
    q=np.std(df[df['clust']==2][df1.index[i]])*1/1.25
    print(q)
    s1[i]=[float(q),0,0,q]
    sz1[i]=q
    
    print(str(i)+" "+df2.index[i])
    q=np.std(df[df['clust']==2][df2.index[i]])*1/1.25
    print(q)
    s2[i]=[float(q),0,0,q]
    sz2[i]=q
    
    print(str(i)+" "+df3.index[i])
    q=np.std(df[df['clust']==3][df3.index[i]])*1/1.25
    print(q)
    s3[i]=[float(q),0,0,q]
    sz3[i]=q

    
    
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.hlines(y=Srt.index, xmin=-10, xmax=10, color='gray', alpha=0.7, 
          linewidth=1, linestyles='dashdot')
ax.scatter(y=Srt.index, x=df0.loc[Srt.index], s=400*np.power(sz0,2), c='green', alpha=0.7,
           label="Cluster 0",)
ax.scatter(y=Srt.index, x=df1.loc[Srt.index], s=400*np.power(sz1,2), c='yellow', alpha=0.7,
           label="Cluster 1")
#ax.scatter(y=Srt.index, x=df2.loc[Srt.index], s=500*np.power(sz2,2), c='orange', alpha=0.7,
#           label="Cluster 2")
#ax.scatter(y=Srt.index, x=df3.loc[Srt.index], s=500*np.power(sz3,2), c='red', alpha=0.7,
#           label="Cluster 3")


ax.vlines(x=0, ymin=0, ymax=len(df0)-1, color='black', alpha=0.7, linewidth=2, linestyles='dotted')
plt.legend(fontsize=20,
           facecolor='White', framealpha=1,frameon=True)

ax.set_title('Mean of Distribution - C18'+' Clusters', fontdict={'size':22})
ax.set_xlim(-4, 1)
plt.savefig("Plots/C18_Means_Clusters.png",dpi=100)
plt.show()    

df=dbak
