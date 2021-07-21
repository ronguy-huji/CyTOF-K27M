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
dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets_Normalized/"

F1="C09-P6-Norm"
#F2="C14"

#aaa=pd.read_csv(dir+"C15-C16_Norm.csv")

df=pd.read_csv(dir+F1+".csv")
dfmut=df
#dfmut=pd.read_csv(dir+F2+"_Norm.csv")

#dfmut_bak=dfmut
Names=['H3',
 'Idu',
 'MBP',
 'H3K36me3',
 'H2B',
 'GFAP',
 'EZH2',
 'H3K4me3',
 'pHistone H2A.X',
 'H3K36me2',
 'Sox2',
 'pYAP',
 'H4K16ac',
 'H2AK119ub',
 'H3K4me1',
 'H3.3',
 'H3K64ac',
 'BMI-1',
 'c-Myc',
 'H4',
 'PDGFRA',
 'YAP',
 'cleaved H3',
 'H3K9ac',
 'CD24',
 'H3K27me3',
 'H3K27M',
 'H3K9me3',
 'Ki-67',
 'pHistone H3 [S28]',
 'H1'
 ]

NamesClust=['H3',
 'Idu',
 'MBP',
 'H3K36me3',
 'H2B',
 'GFAP',
 'EZH2',
 'H3K4me3',
 'pHistone H2A.X',
 'H3K36me2',
 'Sox2',
 'pYAP',
 'H4K16ac',
 'H2AK119ub',
 'H3K4me1',
 'H3.3',
 'H3K64ac',
 'BMI-1',
 'c-Myc',
 'H4',
 'PDGFRA',
 'YAP',
 'cleaved H3',
 'H3K9ac',
 'CD24',
 'H3K27me3',
 'H3K27M',
 'H3K9me3',
 'Ki-67',
 'pHistone H3 [S28]',
 'H1','clust'
 ]


### tSNE Plots
for NN in Names:
    Var=NN
    TSNEVar=NN
    cc=dfmut[TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(dfmut['tsne0'],dfmut['tsne1'],s=2,
                c=cc, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(-3.5,3.5)
    plt.title(TSNEVar)
    plt.savefig("Plots/C09_P6_tSNE_"+NN+".png",dpi=100,bbox_inches = 'tight')


plt.figure(figsize=(6, 5))
plt.scatter(dfmut['tsne0'],dfmut['tsne1'],s=2,
                c=dfmut['clust'], cmap=plt.cm.jet)
plt.clim(-3.5,3.5)
plt.title("Clusters")
plt.colorbar()
plt.savefig("Plots/C09_P6_tSNE_Clust.png",dpi=100,bbox_inches = 'tight')


### All Clusters
dfmut_bak=dfmut

dfmut=dfmut_bak[dfmut_bak['clust'].isin([0,1])]




df=df[Names]
dfmut=dfmut[Names]

plt.figure(figsize=(6, 5))


aaa=dfmut


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
 #'H2A',
 'pHistone H3 [S28]']
Open_Chromatin=[
     'H3K4me1', 'H3K4me3', 'H4K16Ac', 'H3K9ac', 'H3K64ac', 'H3K27ac'
     ]
Closed_Chromatin=['H3K9me3', 'H3K27me3', 'H3K36me2', 'H2AK119ub', 'H3K9ac','H3K27ac']


aaa=dfmut_bak

for NN in NamesClust:
    sns.kdeplot(data=aaa,x=NN,hue='clust',fill=True)
    plt.title(F1+" "+NN+" All Clusters")
    plt.savefig("Plots/"+F1+"_"+NN+"_Hist_All_Clust.png",dpi=100,bbox_inches = 'tight')
    plt.show()    

    


### Cluster 0/1 Comparison


df=dfmut_bak[dfmut_bak['clust'].isin([0])]
dfmut=dfmut_bak[dfmut_bak['clust'].isin([1])]
aaa=df.assign(Type='Cluster 0')
aaa=aaa.append(dfmut.assign(Type='Cluster 1')
    )





df=df[Names]
dfmut=dfmut[Names]

plt.figure(figsize=(6, 5))







dfb=df
sns.set_style({'legend.frameon':True})
df=np.mean(aaa[aaa['Type']=="Cluster 0"][Names]).sort_values()
s1=np.full((len(df),4),0,dtype=np.float64)
sz1=np.full(len(df),0,dtype=np.float64)
df2=np.mean(aaa[aaa['Type']=="Cluster 1"][Names]).sort_values()
s2=np.full((len(df2),4),0,dtype=np.float64)
sz2=np.full(len(df2),0,dtype=np.float64)


Srt=df2
Srt=Srt.sort_values(ascending=False)


for i in range(0,len(df)):
    print(str(i)+" "+df.index[i])
    q=np.std(aaa[aaa['Type']=="Cluster 0"][df.index[i]])*1/1.25
    print(q)
    s1[i]=[float(q),0,0,q]
    sz1[i]=q
    print(str(i)+" "+df2.index[i])
    q=np.std(aaa[aaa['Type']=="Cluster 1"][df2.index[i]])*1./1.25
    print(q)
    s2[i]=[0,0,float(q),q]
    sz2[i]=q
    
    
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.hlines(y=Srt.index, xmin=-5, xmax=5, color='gray', alpha=0.7, 
          linewidth=1, linestyles='dashdot')

#ax.scatter(y=df.index, x=df, s=900*np.power(sz1,2), c='red', alpha=0.7,
#           label="Mutant",)
#ax.scatter(y=df2.index, x=df2, s=900*np.power(sz2,2), c='blue', alpha=0.7,
#           label="Wild")

ax.scatter(y=Srt.index, x=df.loc[Srt.index], s=900*np.power(sz1,2), c='red', alpha=0.7,
           label="Cluster 0",)
ax.scatter(y=Srt.index, x=df2.loc[Srt.index], s=900*np.power(sz2,2), c='blue', alpha=0.7,
           label="Cluster 1")


ax.vlines(x=0, ymin=0, ymax=len(df)-1, color='black', alpha=0.7, linewidth=2, linestyles='dotted')
plt.legend(fontsize=20,
           facecolor='White', framealpha=1,frameon=True)

ax.set_title('Mean of Distribution - C09 P6'+' Clusters', fontdict={'size':22})
ax.set_xlim(-1.5, 2)
plt.savefig("Plots/C09_P6_Means_Clusters.png",dpi=100)
plt.show()

df=dfb

