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
dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets_Normalized_new/"

F1="C13"
F2="C14"

#aaa=pd.read_csv(dir+"C15-C16_Norm.csv")

df=pd.read_csv(dir+F1+"_Norm.csv")
dfmut=pd.read_csv(dir+F2+"_Norm.csv")

dfmut_bak=dfmut

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
 'pHistone H2A.X [Ser139]',
 'pHistone H3 [S28]']



#### tSNE C13
ddf=df[Names]

ar=ddf#np.asarray(ddf)
tsne = TSNE(n_components=2, random_state=0)

X_2d = tsne.fit_transform(ar)#[mask,:])

plt.figure(figsize=(6, 5))
plt.scatter(X_2d[:,0],X_2d[:,1],s=2,
            color='blue')
plt.title("tSNE C13")
plt.savefig("Plots/C13_tSNE.png",dpi=100,bbox_inches = 'tight')

plt.figure(figsize=(6, 5))
plt.scatter(dfmut['tsne0'],dfmut['tsne1'],s=2,
            color='red')
plt.title("tSNE C14")
plt.savefig("Plots/C14_tSNE.png",dpi=100,bbox_inches = 'tight')


ddf=df[Names]
ddf=ddf.append(dfmut[Names])
ar=ddf
X_2d = tsne.fit_transform(ar)#[mask,:])

plt.figure(figsize=(6, 5))
plt.scatter(X_2d[0:len(df),0],X_2d[0:len(df),1],s=2,
            color='blue', label="WT")
plt.scatter(X_2d[len(df):,0],X_2d[len(df):,1],s=2,
            color='red',label="Mutant")
plt.legend(markerscale=5.,frameon=True, fancybox=True)   
plt.title("tSNE C13+C14")
plt.savefig("Plots/C13_C14_tSNE.png",dpi=100,bbox_inches = 'tight')

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
    plt.title("C14 "+TSNEVar)
    plt.savefig("Plots/C14_tSNE_"+NN+".png",dpi=100,bbox_inches = 'tight')


plt.figure(figsize=(6, 5))
plt.scatter(dfmut['tsne0'],dfmut['tsne1'],s=2,
                c=dfmut['clust'], cmap=plt.cm.jet)
plt.clim(-3.5,3.5)
plt.title("Clusters")
plt.colorbar()
plt.savefig("Plots/C14_tSNE_Clust.png",dpi=100,bbox_inches = 'tight')




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

##Cluster Histograms


df=dfmut_bak[dfmut_bak['clust'].isin([0])]
dfmut=dfmut_bak[dfmut_bak['clust'].isin([1])]
aaa=df.assign(Type='Cluster 0')
aaa=aaa.append(dfmut.assign(Type='Cluster 1')
    )

size=min(len(dfmut),len(df))
EqSize=aaa[aaa['Type']=="Cluster 0"].iloc[0:size]
EqSize=EqSize.append(aaa[aaa['Type']=="Cluster 1"].iloc[0:size])

for NN in Names:
    sns.kdeplot(data=EqSize,x=NN,hue='Type',fill=True,palette=['red','blue'])
    plt.title(F2+" Clusters "+NN)
    plt.savefig("Plots/"+F2+"_"+NN+"_Clusters_Hist.png",dpi=100,bbox_inches = 'tight')
    plt.show()    



### Means Plot
Names2=['H2AK119ub',
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
 'pHistone H3 [S28]','Type']



sns.set_style({'legend.frameon':True})
df=np.mean(aaa[aaa['Type']=="Cluster 0"][Names2]).sort_values()
s1=np.full((len(df),4),0,dtype=np.float64)
sz1=np.full(len(df),0,dtype=np.float64)
df2=np.mean(aaa[aaa['Type']=="Cluster 1"][Names2]).sort_values()
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
ax.scatter(y=Srt.index, x=df.loc[Srt.index], s=900*np.power(sz1,2), c='red', alpha=0.7,
           label="Cluster 0",)
ax.scatter(y=Srt.index, x=df2.loc[Srt.index], s=900*np.power(sz2,2), c='blue', alpha=0.7,
           label="Cluster 1")


ax.vlines(x=0, ymin=0, ymax=len(df)-1, color='black', alpha=0.7, linewidth=2, linestyles='dotted')
plt.legend(fontsize=20,
           facecolor='White', framealpha=1,frameon=True)

ax.set_title('Mean of Distribution - C14'+' Clusters', fontdict={'size':22})
ax.set_xlim(-1.5, 2)
plt.savefig("Plots/C14_Means_Clusters.png",dpi=100)
plt.show()    



'''
### All Clusters

dfmut=dfmut_bak[dfmut_bak['clust'].isin([0,1])]
aaa=df.assign(Type='Wild')
aaa=aaa.append(dfmut.assign(Type='Mutant')
    )




df=df[Names]
dfmut=dfmut[Names]

plt.figure(figsize=(6, 5))





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




for NN in Names:
    sns.kdeplot(data=aaa,x=NN,hue='Type',fill=True)
    plt.title(F1+"/"+F2+" "+NN+" All Clusters")
    plt.savefig("Plots/"+F1+"_"+F2+"_"+NN+"_Hist_All_Clust.png",dpi=100,bbox_inches = 'tight')
    plt.show()    

    



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


dfb=df
sns.set_style({'legend.frameon':True})
df=np.mean(aaa[aaa['Type']=="Mutant"][Names]).sort_values()
s1=np.full((len(df),4),0,dtype=np.float64)
sz1=np.full(len(df),0,dtype=np.float64)
df2=np.mean(aaa[aaa['Type']=="Wild"][Names]).sort_values()
s2=np.full((len(df2),4),0,dtype=np.float64)
sz2=np.full(len(df2),0,dtype=np.float64)

for i in range(0,len(df)):
    print(str(i)+" "+df.index[i])
    q=np.std(aaa[aaa['Type']=="Mutant"][df.index[i]])*1/1.25
    print(q)
    s1[i]=[float(q),0,0,q]
    sz1[i]=q
    print(str(i)+" "+df2.index[i])
    q=np.std(aaa[aaa['Type']=="Wild"][df2.index[i]])*1./1.25
    print(q)
    s2[i]=[0,0,float(q),q]
    sz2[i]=q
    
Srt=df-df2
Srt=Srt
    
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.hlines(y=df.index, xmin=-5, xmax=5, color='gray', alpha=0.7, 
          linewidth=1, linestyles='dashdot')
ax.scatter(y=Srt.index, x=df.loc[Srt.index], s=900*np.power(sz1,2), c='red', alpha=0.7,
           label="Mutant",)
#ax.hlines(y=df.index, xmin=df-s1, xmax=df+s1, color='firebrick', alpha=0.7, 
#          linewidth=3, linestyles='dashdot')
ax.scatter(y=Srt.index, x=df2.loc[Srt.index], s=900*np.power(sz2,2), c='blue', alpha=0.7,
           label="Wild")
#ax.hlines(y=df2.index, xmin=df2-s2, xmax=df2+s2, color='blue', alpha=0.7, 
#          linewidth=3, linestyles='dashdot')
ax.vlines(x=0, ymin=0, ymax=len(df)-1, color='black', alpha=0.7, linewidth=2, linestyles='dotted')
plt.legend(fontsize=20,
           facecolor='White', framealpha=1,frameon=True)

ax.set_title('Mean of Distribution - '+F1+'/'+F2+' All Clusters', fontdict={'size':22})
ax.set_xlim(-1.5, 2)
plt.savefig("Plots/"+F1+"_"+F2+"_Means_AllClusters.png",dpi=100)
plt.show()

df=dfb


### Cluster 0



dfmut=dfmut_bak[dfmut_bak['clust'].isin([0])]
aaa=df.assign(Type='Wild')
aaa=aaa.append(dfmut.assign(Type='Mutant')
    )

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



df=df[Names]
dfmut=dfmut[Names]

plt.figure(figsize=(6, 5))





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




for NN in Names:
    sns.kdeplot(data=aaa,x=NN,hue='Type',fill=True)
    plt.title(F1+"/"+F2+" "+NN+" Cluster 0")
    plt.savefig("Plots/"+F1+"_"+F2+"_"+NN+"_Hist_Clust_0.png",dpi=100,bbox_inches = 'tight')
    plt.show()    

    



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


dfb=df
sns.set_style({'legend.frameon':True})
df=np.mean(aaa[aaa['Type']=="Mutant"][Names]).sort_values()
s1=np.full((len(df),4),0,dtype=np.float64)
sz1=np.full(len(df),0,dtype=np.float64)
df2=np.mean(aaa[aaa['Type']=="Wild"][Names]).sort_values()
s2=np.full((len(df2),4),0,dtype=np.float64)
sz2=np.full(len(df2),0,dtype=np.float64)
Srt=df
Srt=Srt.sort_values()

for i in range(0,len(df)):
    print(str(i)+" "+df.index[i])
    q=np.std(aaa[aaa['Type']=="Mutant"][df.index[i]])*1/1.25
    print(q)
    s1[i]=[float(q),0,0,q]
    sz1[i]=q
    print(str(i)+" "+df2.index[i])
    q=np.std(aaa[aaa['Type']=="Wild"][df2.index[i]])*1./1.25
    print(q)
    s2[i]=[0,0,float(q),q]
    sz2[i]=q
    
    
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.hlines(y=Srt.index, xmin=-5, xmax=5, color='gray', alpha=0.7, 
          linewidth=1, linestyles='dashdot')
ax.scatter(y=Srt.index, x=df.loc[Srt.index], s=900*np.power(sz1,2), c='red', alpha=0.7,
           label="Mutant",)
#ax.hlines(y=df.index, xmin=df-s1, xmax=df+s1, color='firebrick', alpha=0.7, 
#          linewidth=3, linestyles='dashdot')
ax.scatter(y=Srt.index, x=df2.loc[Srt.index], s=900*np.power(sz2,2), c='blue', alpha=0.7,
           label="Wild")
#ax.hlines(y=df2.index, xmin=df2-s2, xmax=df2+s2, color='blue', alpha=0.7, 
#          linewidth=3, linestyles='dashdot')
ax.vlines(x=0, ymin=0, ymax=len(df)-1, color='black', alpha=0.7, linewidth=2, linestyles='dotted')
plt.legend(fontsize=20,
           facecolor='White', framealpha=1,frameon=True)

ax.set_title('Mean of Distribution - '+F1+'/'+F2+' Cluster 0', fontdict={'size':22})
ax.set_xlim(-1.5, 2)
plt.savefig("Plots/"+F1+"_"+F2+"_Means_Cluster0.png",dpi=100)
plt.show()

df=dfb


### Cluster 1



dfmut=dfmut_bak[dfmut_bak['clust'].isin([1])]
aaa=df.assign(Type='Wild')
aaa=aaa.append(dfmut.assign(Type='Mutant')
    )

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



df=df[Names]
dfmut=dfmut[Names]

plt.figure(figsize=(6, 5))





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




for NN in Names:
    sns.kdeplot(data=aaa,x=NN,hue='Type',fill=True)
    plt.title(F1+"/"+F2+" "+NN+" Cluster 1")
    plt.savefig("Plots/"+F1+"_"+F2+"_"+NN+"_Hist_Clust_1.png",dpi=100,bbox_inches = 'tight')
    plt.show()    

    



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


dfb=df
sns.set_style({'legend.frameon':True})
df=np.mean(aaa[aaa['Type']=="Mutant"][Names]).sort_values()
s1=np.full((len(df),4),0,dtype=np.float64)
sz1=np.full(len(df),0,dtype=np.float64)
df2=np.mean(aaa[aaa['Type']=="Wild"][Names]).sort_values()
s2=np.full((len(df2),4),0,dtype=np.float64)
sz2=np.full(len(df2),0,dtype=np.float64)

for i in range(0,len(df)):
    print(str(i)+" "+df.index[i])
    q=np.std(aaa[aaa['Type']=="Mutant"][df.index[i]])*1/1.25
    print(q)
    s1[i]=[float(q),0,0,q]
    sz1[i]=q
    print(str(i)+" "+df2.index[i])
    q=np.std(aaa[aaa['Type']=="Wild"][df2.index[i]])*1./1.25
    print(q)
    s2[i]=[0,0,float(q),q]
    sz2[i]=q
    
    
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.hlines(y=df.index, xmin=-5, xmax=5, color='gray', alpha=0.7, 
          linewidth=1, linestyles='dashdot')
ax.scatter(y=df.index, x=df, s=900*np.power(sz1,2), c='red', alpha=0.7,
           label="Mutant",)
#ax.hlines(y=df.index, xmin=df-s1, xmax=df+s1, color='firebrick', alpha=0.7, 
#          linewidth=3, linestyles='dashdot')
ax.scatter(y=df2.index, x=df2, s=900*np.power(sz2,2), c='blue', alpha=0.7,
           label="Wild")
#ax.hlines(y=df2.index, xmin=df2-s2, xmax=df2+s2, color='blue', alpha=0.7, 
#          linewidth=3, linestyles='dashdot')
ax.vlines(x=0, ymin=0, ymax=len(df)-1, color='black', alpha=0.7, linewidth=2, linestyles='dotted')
plt.legend(fontsize=20,
           facecolor='White', framealpha=1,frameon=True)

ax.set_title('Mean of Distribution - '+F1+'/'+F2+' Cluster 1', fontdict={'size':22})
ax.set_xlim(-1.5, 2)
plt.savefig("Plots/"+F1+"_"+F2+"_Means_Cluster1.png",dpi=100)
plt.show()

df=dfb

### Cluster 0/1 Comparison


df=dfmut_bak[dfmut_bak['clust'].isin([0])]
dfmut=dfmut_bak[dfmut_bak['clust'].isin([1])]
aaa=df.assign(Type='Cluster 0')
aaa=aaa.append(dfmut.assign(Type='Cluster 1')
    )

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



df=df[Names]
dfmut=dfmut[Names]

plt.figure(figsize=(6, 5))





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




for NN in Names:
    sns.kdeplot(data=aaa,x=NN,hue='Type',fill=True)
    plt.title("C14 Clusters")
    plt.savefig("Plots/C14_Hist_Clust_Comp.png",dpi=100,bbox_inches = 'tight')
    plt.show()    

    



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

ax.set_title('Mean of Distribution - C14'+' Clusters', fontdict={'size':22})
ax.set_xlim(-1.5, 2)
plt.savefig("Plots/C4_Means_Clusters.png",dpi=100)
plt.show()
'''

'''
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
ax.hlines(y=df.index, xmin=-5, xmax=5, color='gray', alpha=0.7, 
          linewidth=1, linestyles='dashdot')
ax.scatter(y=df.index, x=df, s=900*np.power(sz1,2), c='red', alpha=0.7,
           label="Cluster 0",)
#ax.hlines(y=df.index, xmin=df-s1, xmax=df+s1, color='firebrick', alpha=0.7, 
#          linewidth=3, linestyles='dashdot')
ax.scatter(y=df2.index, x=df2, s=900*np.power(sz2,2), c='blue', alpha=0.7,
           label="Cluster 1")
#ax.hlines(y=df2.index, xmin=df2-s2, xmax=df2+s2, color='blue', alpha=0.7, 
#          linewidth=3, linestyles='dashdot')
ax.vlines(x=0, ymin=0, ymax=len(df)-1, color='black', alpha=0.7, linewidth=2, linestyles='dotted')
plt.legend(fontsize=20,
           facecolor='White', framealpha=1,frameon=True)

ax.set_title('Mean of Distribution - C14'+' Clusters', fontdict={'size':22})
ax.set_xlim(-1.5, 2)
plt.savefig("Plots/C4_Means_Clusters.png",dpi=100)
plt.show()
'''
df=dfb

