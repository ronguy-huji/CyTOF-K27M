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
import matplotlib
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

ExpName="C17"

dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets/17321/"
C11=pd.read_csv(dir+ExpName+"_Norm_UMAP_2Clust.csv")

C11=C11[C11['clust'].isin([0,1,2,3,4])]



Names=['H3',
 'IdU',
 'MBP',
 'H3K36me3',
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
 'pH3[S28]',
 'H1.3/4/5']

EpiGenList=['H3',
 'H3K36me3',
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
 'H3K27ac',
 'H4K20me3',
 'cleaved H3',
 'H3K9ac',
 'H1.0',
 'H3K27me3',
 'H3K27M',
 'H3K9me3',
 'pH3[S28]',
 'H1.3/4/5']

NamesOther = [ elem for elem in Names if elem not in EpiGenList]



dfmask=pd.read_csv(dir+ExpName+"_Mask.csv")[Names]
#C11[dfmask]=float("nan")


import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

NClust=2

#dfmask=C11==-100
#C11[dfmask]=float("nan")

aaa=C11[C11['clust'].isin([0,1,2,3,4])]
plt.figure(figsize=(6, 5))

for NN in Names:
     BoxVar=NN
     plt.figure(figsize=(3, 5))    
     ax = sns.boxplot(x="clust", y=NN, data=aaa,showfliers=False,palette=['#FF8C00','dimgray'],saturation=1)
     ax.set_xticklabels(['H3K27M \nHigh','H3K27M \nLow'])
     ylims=np.round(ax.get_ylim())
     ymin=np.int(ylims[0])
     ymax=np.int(ylims[1]+1)     
     ax.set_yticks(np.arange(ymin,ymax))
     ax.set_yticklabels(np.arange(ymin,ymax))
     if (NN!="H1.3/4/5"):
         fname="/Users/ronguy/Dropbox/Work/CyTOF/Plots/"+ExpName+"_"+NN+"_Box.pdf"
     else:
         fname="/Users/ronguy/Dropbox/Work/CyTOF/Plots/"+ExpName+"_"+"H1.345_"+"Box.pdf"
     plt.savefig(fname,bbox_inches='tight')     
     plt.show()


'''

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
    
'''
for NN in Names:
    Var=NN
    TSNEVar=NN
    cc=C11[TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(C11['umap0'],C11['umap1'],s=2,
                c=cc, cmap=plt.cm.seismic)
    plt.colorbar()
    plt.clim(-3.5,3.5)
    plt.clim(cc.quantile(0.05),cc.quantile(0.95))
    cmap = matplotlib.cm.get_cmap('jet')



    rgba = cmap(-10)

    mask=dfmask[TSNEVar]
    plt.scatter(C11[mask]['umap0'],C11[mask]['umap1'],s=2,
           color=rgba)
    plt.title(TSNEVar+" "+ExpName+" Epigen")
    
    if (NN!="H1.3/4/5"):
        fname="/Users/ronguy/Dropbox/Work/CyTOF/Plots/UMAP"+ExpName+"_"+NN+".png"
    else:
        fname="/Users/ronguy/Dropbox/Work/CyTOF/Plots/UMAP"+ExpName+"_"+"H1.345.png"

    plt.savefig(fname,
                dpi=100,bbox_inches = 'tight')
    plt.show()
    
    
cList=['blue','red','green','magenta','pink']
plt.figure(figsize=(6,5))

mask=C11['clust']!=1    
plt.scatter(C11[mask]['umap0'],C11[mask]['umap1'],s=2,
            color='darkorange',label="H3K27M High")    
mask=C11['clust']==1    
plt.scatter(C11[mask]['umap0'],C11[mask]['umap1'],s=2,
            color='dimgray',label="H3K27M Low")    
plt.legend(markerscale=5);
plt.title(ExpName+" Clusters Epigen");
plt.savefig("/Users/ronguy/Dropbox/Work/CyTOF/Plots/"+ExpName+"_ClustUMAP.pdf",bbox_inches='tight')     
plt.savefig("/Users/ronguy/Dropbox/Work/CyTOF/Plots/"+ExpName+"_ClustUMAP.png",bbox_inches='tight') 
plt.show()

 

#C11[C11['clust'].isin([0,1,3])]=0

'''

### Mean of Dist ExpName


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


'''
#### Diverging Bars
sns.set_style({'legend.frameon':True})






#### Low vs High 
###EpiGen
sns.set_style({'legend.frameon':True})

d0=C11[C11['clust']!=1]
d1=C11[C11['clust']==1]
d2=C11[C11['clust']==2]
d3=C11[C11['clust']==3]


#d0=d0.drop(columns=['H3','H3.3','H4'])
#d1=d1.drop(columns=['H3','H3.3','H4'])


dd0=np.mean(d0[EpiGenList]).sort_values(ascending=False)
dd1=np.mean(d1[EpiGenList]).sort_values()

dd0=dd0.drop(labels=['H3','H3.3','H4'])
dd1=dd1.drop(labels=['H3','H3.3','H4'])



diffs=(dd0-dd1).sort_values(ascending=False)    
 
colors = ['dimgray' if x < 0 else 'darkorange' for x in diffs]


'''
# Draw plot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=0.4, linewidth=5)

# Decorations
plt.gca().set(ylabel='$Marker$', xlabel='$Difference$')
#plt.yticks(df.index, df.cars, fontsize=12)
plt.title(ExpName+' K27M High vs Low Diffs Epigenetic', fontdict={'size':20})
plt.grid(linestyle='--', alpha=0.5)
plt.show()
'''
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=1, linewidth=5)

# Decorations
plt.gca().set(ylabel='', xlabel='')
plt.xticks(fontsize=30 ) 
plt.yticks(fontsize=30 ) 
#plt.yticks(df.index, df.cars, fontsize=12)
plt.title(ExpName+' K27M High vs Low Diffs Epigenetic', fontdict={'size':20})
plt.grid(linestyle='--', alpha=0.5)
plt.figure(figsize=(6,5))

mask=C11['clust']!=1    
plt.scatter(C11[mask]['umap0'],C11[mask]['umap1'],s=2,
            color='darkorange',label="H3K27M High")    
mask=C11['clust']==1    
plt.scatter(C11[mask]['umap0'],C11[mask]['umap1'],s=2,
            color='dimgray',label="H3K27M Low")    
plt.legend(markerscale=5);
plt.title(ExpName+" Clusters Epigen");
plt.savefig("/Users/ronguy/Dropbox/Work/CyTOF/Plots/"+ExpName+"_Diff_Epi.pdf",bbox_inches='tight')     
plt.show()

plt.show()


### Other


sns.set_style({'legend.frameon':True})

d0=C11[C11['clust']!=1]
d1=C11[C11['clust']==1]
d2=C11[C11['clust']==2]
d3=C11[C11['clust']==3]


#d0=d0.drop(columns=['H3','H3.3','H4'])
#d1=d1.drop(columns=['H3','H3.3','H4'])


dd0=np.mean(d0[NamesOther]).sort_values(ascending=False)
dd1=np.mean(d1[NamesOther]).sort_values()

# dd0=dd0.drop(labels=['H3','H3.3','H4'])
# dd1=dd1.drop(labels=['H3','H3.3','H4'])



diffs=(dd0-dd1).sort_values(ascending=False)    
 
colors = ['dimgray' if x < 0 else 'darkorange' for x in diffs]


'''
# Draw plot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=0.4, linewidth=5)

# Decorations
plt.gca().set(ylabel='$Marker$', xlabel='$Difference$')
#plt.yticks(df.index, df.cars, fontsize=12)
plt.title(ExpName+' K27M High vs Low Diffs Epigenetic', fontdict={'size':20})
plt.grid(linestyle='--', alpha=0.5)
plt.show()
'''
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=1, linewidth=5)

# Decorations
plt.gca().set(ylabel='', xlabel='')
plt.xticks(fontsize=30 ) 
plt.yticks(fontsize=30 ) 
#plt.yticks(df.index, df.cars, fontsize=12)
plt.title(ExpName+' K27M High vs Low Diffs Rest', fontdict={'size':20})
plt.grid(linestyle='--', alpha=0.5)
plt.savefig("/Users/ronguy/Dropbox/Work/CyTOF/Plots/"+ExpName+"_Diff_Rest.pdf",bbox_inches='tight')  
plt.show()




#### Bar Graphs
df=pd.read_csv(dir+ExpName+"_Mask.csv")

df=df[df['clust'].isin([0,1])]

ColNames=list(df.columns)
A=df.groupby('clust').sum().drop(columns=['Unnamed: 0'])

for clust in A.index.values:
    lenC=(C11['clust']==clust).sum()
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
for NN in Names:
    mask=B['variable']==NN
    fig=plt.figure(dpi= 380, figsize=(3,5))
    ax = fig.add_axes([0,0,1,1])
    Clust = ['H3K27M \nHigh', 'H3K27M \nLow']
    Vals = B[mask].value
    ax.bar(Clust,Vals,color=['darkorange','dimgray'])
    plt.title(ExpName+" " + NN)
    if (NN!="H1.3/4/5"):
        fname="/Users/ronguy/Dropbox/Work/CyTOF/Plots/"+ExpName+"_"+NN+"_Frac.pdf"
    else:
        fname="/Users/ronguy/Dropbox/Work/CyTOF/Plots/"+ExpName+"_"+"H1.345_"+"Frac.pdf"
    plt.savefig(fname,bbox_inches='tight')  
    plt.show()


# plt.figure(figsize=(10,10))
# g=sns.kdeplot(data=C11,x='Sox2',y='Ki-67',levels=10,hue='clust',palette=("darkorange","dimgray"),common_norm=False)
# g.legend_.set_title("Cluster")
# new_labels = ['H3K27M High', 'H3K27M Low']
# for t, l in zip(g.legend_.texts, new_labels): t.set_text(l)
# plt.title(ExpName)
# plt.show()

# plt.figure(figsize=(10,10))
# g=sns.kdeplot(data=C11,x='Sox2',y='IdU',levels=10,hue='clust',palette=("darkorange","dimgray"),common_norm=False)
# g.legend_.set_title("Cluster")
# new_labels = ['H3K27M High', 'H3K27M Low']
# for t, l in zip(g.legend_.texts, new_labels): t.set_text(l)
# plt.title(ExpName)
# plt.show()

# plt.figure(figsize=(10,10))
# g=sns.kdeplot(data=C11,x='Cmyc',y='Ki-67',levels=10,hue='clust',palette=("darkorange","dimgray"),common_norm=False)
# g.legend_.set_title("Cluster")
# new_labels = ['H3K27M High', 'H3K27M Low']
# for t, l in zip(g.legend_.texts, new_labels): t.set_text(l)
# plt.title(ExpName)
# plt.show()

# plt.figure(figsize=(10,10))
# g=sns.kdeplot(data=C11,x='Cmyc',y='IdU',levels=10,hue='clust',palette=("darkorange","dimgray"),common_norm=False)
# g.legend_.set_title("Cluster")
# new_labels = ['H3K27M High', 'H3K27M Low']
# for t, l in zip(g.legend_.texts, new_labels): t.set_text(l)
# plt.title(ExpName)
# plt.show()


C11[dfmask]=float("nan")

Open_Chromatin=[
     'H3K4me1', 'H3K4me3', 'H4K16ac', 'H3K9ac', 'H3K64ac', 'H3K27ac'
     ]
Closed_Chromatin=['H3K9me3', 'H3K27me3', 'H3K36me2', 'H2Aub', 'H3K9ac','H3K27ac']

sns.set(font_scale=1.2)

params = {'axes.titlesize': 30,
          'legend.fontsize': 20,
          'figure.figsize': (16, 10),
          'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'figure.titlesize': 30}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")


print("Cluster map "+ExpName)
plt.figure(figsize=(10,10))
matrix=aaa[EpiGenList].corr()
g=sns.clustermap(matrix, annot=True,
        cmap=plt.cm.jet,vmin=-.2,vmax=1,linewidths=.1); 
plt.xticks(rotation=90); 
plt.yticks(rotation=0); 
plt.title(ExpName+' Epigen')
plt.show()

matrix.to_excel("Corr_EpiGen_"+ExpName+".xls")


print("Cluster map "+ExpName)
plt.figure(figsize=(10,10))
matrix=C11[Open_Chromatin].corr()
g=sns.clustermap(matrix, annot=True, annot_kws={"size":20},
        cmap=plt.cm.jet,vmin=-.2,vmax=1,linewidths=.1); 
plt.xticks(rotation=90); 
plt.yticks(rotation=0); 
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
#plt.title(ExpName+' Open Chromatin')
plt.savefig("/Users/ronguy/Dropbox/Work/CyTOF/Plots/"+ExpName+"_OpenCh.pdf",bbox_inches = 'tight')
plt.show()


plt.figure(figsize=(10,10))
matrix=C11[Closed_Chromatin].corr()
g=sns.clustermap(matrix, annot=True, annot_kws={"size":20},
        cmap=plt.cm.jet,vmin=-.2,vmax=1,linewidths=.1); 
plt.xticks(rotation=90); 
plt.yticks(rotation=0); 
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
#plt.title(ExpName+' Closed Chromatin')
plt.savefig("/Users/ronguy/Dropbox/Work/CyTOF/Plots/"+ExpName+"_ClosedCh.pdf",bbox_inches = 'tight')
plt.show()