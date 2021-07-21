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

import ppscore as pps


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
dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets/Nada/"

F1="C07"
F2="C06"

#aaa=pd.read_csv(dir+"C05-C07_Norm.csv")

df=pd.read_csv(dir+F1+"_Nada.csv")
dfmut=pd.read_csv(dir+F2+"_Nada.csv")

#df=aaa[aaa['Type']=="Wild"]
#dfmut=aaa[aaa['Type']=="Mutant"]
#df=df.assign(Type='WT')
#dfmut=dfmut.assign(Type='Mutant')

NamesAll=['H3',
 'IdU',
 'MBP',
 'H3K36me3',
 'H2B',
 'GFAP',
 'EZH2',
 'H3K4me3',
 'yH2A.X',
 'H3K36me2',
 'Sox2',
 'pYAP',
 'H4K16ac',
 'H2Aub',
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
 'pH3[S28]',
 'H1']

NamesUMAP=['H3',
 'H3K36me3',
 'H3K4me3',
 'H3K36me2',
 'H4K16ac',
 'H2Aub',
 'H3K4me1',
 'H3.3',
 'H3K64ac',
 'H4',
 'cleaved H3',
 'H3K9ac',
 'H3K27me3',
 'H3K27M',
 'H3K9me3',]


GateColumns=['H3.3','H4','H3']

            
df=df[(df[GateColumns]>5).all(axis=1)]
dfmut=dfmut[(dfmut[GateColumns]>5).all(axis=1)]


GateColumns=['H3',
 'H3K36me3',
 'H3K4me3',
 'H3K36me2',
 'H4K16ac',
 'H2Aub',
 'H3K4me1',
 'H3.3',
 'H3K64ac',
 'H4',
 'cleaved H3',
 'H3K9ac',
 'H3K27me3',
 'H3K27M',
 'H3K9me3',]

df=df[(df[GateColumns]>0).all(axis=1)]
dfmut=dfmut[(dfmut[GateColumns]>0).all(axis=1)]



scFac=5
df=np.arcsinh(df/scFac)
dfmut=np.arcsinh(dfmut/scFac)




aaaa=df.append(dfmut).copy()
m=np.mean(aaaa)
s=np.std(aaaa)


dfBCK=((df-m)/s).copy()
dfmutBCK=((dfmut-m)/s).copy()


aaa=df
aaa=aaa.append(dfmut)


EqualWeights=0
if EqualWeights==0:
    params = Parameters()
    params.add('alpha', value=0.5, min=0)
    params.add('beta', value=0.5, min=0)
    params.add('gamma', value=0.5, min=0)
    aaaa=(aaaa-m)/s
    out = minimize(residual, params, args=(aaaa, aaaa),method='cg')
    alpha=out.params['alpha'].value
    beta=out.params['beta'].value
    gam=out.params['gamma'].value
    
    ddf=df
    ddf=(ddf-m)/s
    df=ddf.copy()
    out = minimize(residual, params, args=(ddf, ddf),method='cg')
    alpha=out.params['alpha'].value
    beta=out.params['beta'].value
    gam=out.params['gamma'].value
    avMarkers=ddf['H3.3']*alpha+ddf['H4']*beta+ddf['H3']*gam
    ddf=ddf.subtract(avMarkers,axis=0)
    df=ddf
    
    ddf=dfmut
    ddf=(ddf-m)/s
    dfmut=ddf.copy()
    out = minimize(residual, params, args=(ddf, ddf),method='cg')
    alpha=out.params['alpha'].value
    beta=out.params['beta'].value
    gam=out.params['gamma'].value
    avMarkers=ddf['H3.3']*alpha+ddf['H4']*beta+ddf['H3']*gam
    ddf=ddf.subtract(avMarkers,axis=0)
    dfmut=ddf    

dfBCK[GateColumns]=df[GateColumns]
dfmutBCK[GateColumns]=dfmut[GateColumns]


df=dfBCK.copy()
dfmut=dfmutBCK.copy()

df=df.assign(Type='WT')
dfmut=dfmut.assign(Type='Mutant')



CAll=df.append(dfmut)







X_2d=draw_umap(CAll[NamesUMAP],cc=CAll['H3K27M'],min_dist=0.05)

for NN in NamesAll:
    Var=NN
    TSNEVar=NN
    cc=CAll[TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:,0],X_2d[:,1],s=2,
                c=cc, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(-3.5,3.5)
#    cmap = matplotlib.cm.get_cmap('jet')
#    mask=dfmask[TSNEVar]==True
#    rgba = cmap(-10)
#    plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],s=2,
#                color=rgba) 
    plt.title(TSNEVar)
    
plt.figure(figsize=(6, 5))
mask=CAll.Type=='WT'
plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],s=2,
                color='blue',label='WT')
mask=CAll.Type=='Mutant'
plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],s=2,
                color='red',label='Mutant')
plt.legend(markerscale=5)
plt.title("C06/C07")
plt.show() 


for NN in NamesAll:
     BoxVar=NN
     plt.figure(figsize=(3, 5))    
     ax = sns.boxplot(x="Type", y=NN, data=CAll,showfliers=False,palette=['red','blue','green'])
     plt.title(NN+" Nada")
     if (NN!="H1.3/4/5"):
         fname="/Users/ronguy/Dropbox/Work/CyTOF/Plots/"+"Nada"+"_"+NN+"_Box.pdf"
     else:
         fname="/Users/ronguy/Dropbox/Work/CyTOF/Plots/"+"Nada"+"_"+"H1.345_"+"Box.pdf"
     plt.savefig(fname,bbox_inches='tight')     
     plt.show()   

sns.set_style({'legend.frameon':True})
#CAll=CAll[CAll['clust'].isin([0,1])]
d0=df.copy()
d1=dfmut.copy()


dd0=d0[NamesAll].mean().sort_values(ascending=False)
dd1=d1[NamesAll].mean().sort_values()



diffs=(dd1-dd0).sort_values(ascending=False)    
 
colors = ['blue' if x < 0 else 'red' for x in diffs]



# Draw plot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=1, linewidth=5)

# Decorations
plt.gca().set(ylabel='', xlabel='')
plt.xticks(fontsize=20 ) 
plt.yticks(fontsize=20 ) 
#plt.yticks(df.index, df.cars, fontsize=12)
plt.title('Diffs Nada', fontdict={'size':20})
plt.grid(linestyle='--', alpha=0.5)
plt.show()






# Names=['H2Aub',
#  'H2B',
#  'H3',
#  'H3.3',
# # 'H3K27ac',
#  'H3K27M',
#  'H3K27me3',
#  'H3K36me2',
#  'H3K36me3',
#  'H3K4me1',
#  'H3K4me3',
#  'H3K64ac',
#  'H3K9ac',
#  'H3K9me3',
#  'H4',
#  'H4K16ac',
#  'cleaved H3',
#  #'H2A',
#  'pH3[S28]']



# df=df[Names]
# dfmut=dfmut[Names]

# plt.figure(figsize=(6, 5))





# Names_All=['H2Aub',
# # 'H3K27ac',
#  'H3K27me3',
#  'H3K36me2',
#  'H3K36me3',
#  'H3K4me1',
#  'H3K4me3',
#  'H3K64ac',
#  'H3K9ac',
#  'H3K9me3',
#  'H4K16ac',
#  'cleaved H3',
#  #'H2A',
#  'pH3[S28]']
# Open_Chromatin=[
#      'H3K4me1', 'H3K4me3', 'H4K16ac', 'H3K9ac', 'H3K64ac'
#      ]
# Closed_Chromatin=['H3K9me3', 'H3K27me3', 'H3K36me2', 'H2Aub', 'H3K9ac']


# ### Heatmaps for wild 

# #sns.heatmap(dfmut[Names].corr()-df[Names].corr(),annot=True,cmap=plt.cm.jet,vmin=0,vmax=1,linewidths=.1);
# plt.figure(figsize=(10, 10))
# g=sns.clustermap(df[Names_All].corr(),
#                  annot=True,
#                  cmap=plt.cm.jet,
#                  vmin=0,vmax=1,linewidths=.1,
#                  cbar_pos=(1, 0.5, 0.05, 0.18));
# plt.xticks(rotation=90); 
# plt.yticks(rotation=0)
# g.ax_col_dendrogram.set_visible(False)
# #g.ax_col_dendrogram.set_xlim([0,0])
# #g.ax_cbar.set_position((0.8, .2, .03, .4))
# plt.title(F1+" All"); 
# plt.savefig("Plots/"+F1+"_All.png",dpi=100,bbox_inches = 'tight')
# plt.show()




# plt.figure(figsize=(10, 10))
# g=sns.clustermap(df[Open_Chromatin].corr(),
#                  annot=True,
#                  cmap=plt.cm.jet,
#                  vmin=0,vmax=1,linewidths=.1,
#                  cbar_pos=(1, 0.5, 0.05, 0.18));
# plt.xticks(rotation=90); 
# plt.yticks(rotation=0)
# g.ax_col_dendrogram.set_visible(False)
# #g.ax_col_dendrogram.set_xlim([0,0])
# #g.ax_cbar.set_position((0.8, .2, .03, .4))
# plt.title(F1+" Open Chromatin");
# plt.savefig("Plots/"+F1+"_Open.png",dpi=100,bbox_inches = 'tight') 
# plt.show()


# plt.figure(figsize=(10, 10))
# g=sns.clustermap(df[Closed_Chromatin].corr(),
#                  annot=True,
#                  cmap=plt.cm.jet,
#                  vmin=0,vmax=1,linewidths=.1,
#                  cbar_pos=(1, 0.5, 0.05, 0.18));
# plt.xticks(rotation=90); 
# plt.yticks(rotation=0)
# g.ax_col_dendrogram.set_visible(False)
# #g.ax_col_dendrogram.set_xlim([0,0])
# #g.ax_cbar.set_position((0.8, .2, .03, .4))
# plt.title(F1+" Closed Chromatin"); 
# plt.savefig("Plots/"+F1+"_Closed.png",dpi=100,bbox_inches = 'tight')
# plt.show()
# #### Heatmaps for Mutant

# #sns.heatmap(dfmut[Names].corr()-df[Names].corr(),annot=True,cmap=plt.cm.jet,vmin=0,vmax=1,linewidths=.1);
# plt.figure(figsize=(10, 10))
# g=sns.clustermap(dfmut[Names_All].corr(),
#                  annot=True,
#                  cmap=plt.cm.jet,
#                  vmin=0,vmax=1,linewidths=.1,
#                  cbar_pos=(1, 0.5, 0.05, 0.18));
# plt.xticks(rotation=90); 
# plt.yticks(rotation=0)
# g.ax_col_dendrogram.set_visible(False)
# #g.ax_col_dendrogram.set_xlim([0,0])
# #g.ax_cbar.set_position((0.8, .2, .03, .4))
# plt.title(F2+" All"); 
# plt.savefig("Plots/"+F2+"_All.png",dpi=100,bbox_inches = 'tight')
# plt.show()




# plt.figure(figsize=(10, 10))
# g=sns.clustermap(dfmut[Open_Chromatin].corr(),
#                  annot=True,
#                  cmap=plt.cm.jet,
#                  vmin=0,vmax=1,linewidths=.1,
#                  cbar_pos=(1, 0.5, 0.05, 0.18));
# plt.xticks(rotation=90); 
# plt.yticks(rotation=0)
# g.ax_col_dendrogram.set_visible(False)
# #g.ax_col_dendrogram.set_xlim([0,0])
# #g.ax_cbar.set_position((0.8, .2, .03, .4))
# plt.title(F2+" Open Chromatin");
# plt.savefig("Plots/"+F2+"_Open.png",dpi=100,bbox_inches = 'tight') 
# plt.show()


# plt.figure(figsize=(10, 10))
# g=sns.clustermap(dfmut[Closed_Chromatin].corr(),
#                  annot=True,
#                  cmap=plt.cm.jet,
#                  vmin=0,vmax=1,linewidths=.1,
#                  cbar_pos=(1, 0.5, 0.05, 0.18));
# plt.xticks(rotation=90); 
# plt.yticks(rotation=0)
# g.ax_col_dendrogram.set_visible(False)
# #g.ax_col_dendrogram.set_xlim([0,0])
# #g.ax_cbar.set_position((0.8, .2, .03, .4))
# plt.title(F2+" Closed Chromatin"); 
# plt.savefig("Plots/"+F2+"_Closed.png",dpi=100,bbox_inches = 'tight')
# plt.show()
# ####
# #### Delta correlations 


# plt.figure(figsize=(10, 10))
# g=sns.clustermap(dfmut[Names_All].corr()-df[Names_All].corr(),
#                  annot=True,
#                  cmap=plt.cm.jet,
#                  vmin=0,vmax=1,linewidths=.1,
#                  cbar_pos=(1, 0.5, 0.05, 0.18));
# plt.xticks(rotation=90); 
# plt.yticks(rotation=0)
# g.ax_col_dendrogram.set_visible(False)
# #g.ax_col_dendrogram.set_xlim([0,0])
# #g.ax_cbar.set_position((0.8, .2, .03, .4))
# plt.title(F1+"/"+F2+" All Diff"); 
# plt.savefig("Plots/"+F1+"_"+F2+"_Diff.png",dpi=100,bbox_inches = 'tight')
# plt.show()



# ####

# # for NN in Names:
# #     sns.kdeplot(data=aaa,x=NN,hue='Type',fill=True)
# #     plt.title(F1+"/"+F2+" "+NN)
# #     plt.savefig("Plots/"+F1+"_"+F2+"_"+NN+"_Hist.png",dpi=100,bbox_inches = 'tight')
# #     plt.show()    

    

# '''

# title='C15'

# Vars=['H3K27ac','H3K9ac','H3K64ac','H4K16Ac']
# plt.figure(figsize=(6, 5))    
# sns.pairplot(df[Vars],corner=True,plot_kws={"s": 3})
# plt.title(title)

# plt.figure(figsize=(6, 5))    
# sns.heatmap(dfmut[Vars].corr(),annot=True,cmap=plt.cm.jet,vmin=0,vmax=1,linewidths=.1); 
# plt.xticks(rotation=90); 
# plt.yticks(rotation=0); 
# plt.show()
# '''

# '''

# Vars=['H3K4me3','H3K4me1','H4K16Ac','H3K9ac','H3K64ac','H3K27ac']
# plt.figure(figsize=(6, 5)) 
# sns.pairplot(df[Vars],corner=True,plot_kws={"s": 3})
# plt.title(title)

# ### SAME for C13

# Vars=['H3K9me3','H3K27me3', 'H3K36me2', 'H2AK119ub', 'H3K9ac', 'H3K27ac']
# plt.figure(figsize=(6, 5))    
# sns.pairplot(df[Vars],corner=True,plot_kws={"s": 3})
# plt.title(title)

# '''

# '''
# ## SAME for C13
# size=min(len(dfmut),len(df))
# EqSize=aaaa[aaaa['Type']=="Mutant"].iloc[0:size]
# EqSize=EqSize.append(aaaa[aaaa['Type']=="Wild"].iloc[0:size])


# title='C15/C16'
# Vars=['H3K27ac','H4K16Ac','H3K4me1','H3K4me3','H2AK119ub','H3K27me3','Type']
# plt.figure(figsize=(6, 5))    
# sns.pairplot(EqSize[Vars].sort_values("Type",ascending=False),
#              corner=True,hue="Type",plot_kws={"s": 3},
#         )
# plt.title(title)

# '''

# '''
# Vars=['H3K27ac','H4K16Ac','H3K4me1','H3K4me3','H2AK119ub','H3K27me3']
# plt.figure(figsize=(6, 5))    
# sns.pairplot(dfmut[Vars],corner=True,plot_kws={"s": 3})
# plt.title("C16")
# '''

# Names=['H2Aub',
#  'H2B',
#  'H3',
#  'H3.3',
#  'H3K27M',
#  'H3K27me3',
#  'H3K36me2',
#  'H3K36me3',
#  'H3K4me1',
#  'H3K4me3',
#  'H3K64ac',
#  'H3K9ac',
#  'H3K9me3',
#  'H4',
#  'H4K16ac',
#  'cleaved H3',
#  #'H2A',
#  'pH3[S28]']

# dfb=df
# sns.set_style({'legend.frameon':True})
# df=np.mean(aaa[aaa['Type']=="Mutant"][Names]).sort_values()
# s1=np.full((len(df),4),0,dtype=np.float64)
# sz1=np.full(len(df),0,dtype=np.float64)
# df2=np.mean(aaa[aaa['Type']=="WT"][Names]).sort_values()
# s2=np.full((len(df2),4),0,dtype=np.float64)
# sz2=np.full(len(df2),0,dtype=np.float64)

# Srt=df-df2
# Srt=Srt.sort_values(ascending=False)


# for i in range(0,len(df)):
#     print(str(i)+" "+df.index[i])
#     q=np.std(aaa[aaa['Type']=="Mutant"][df.index[i]])*1/1.25
#     print(q)
#     s1[i]=[float(q),0,0,q]
#     sz1[i]=q
#     print(str(i)+" "+df2.index[i])
#     q=np.std(aaa[aaa['Type']=="WT"][df2.index[i]])*1./1.25
#     print(q)
#     s2[i]=[0,0,float(q),q]
#     sz2[i]=q
    
    
# fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
# ax.hlines(y=df.index, xmin=-5, xmax=5, color='gray', alpha=0.7, 
#           linewidth=1, linestyles='dashdot')

# #ax.scatter(y=df.index, x=df, s=900*np.power(sz1,2), c='red', alpha=0.7,
# #           label="Mutant",)
# #ax.scatter(y=df2.index, x=df2, s=900*np.power(sz2,2), c='blue', alpha=0.7,
# #           label="Wild")

# ax.scatter(y=Srt.index, x=df.loc[Srt.index], s=900*np.power(sz1,2), c='red', alpha=0.7,
#            label="Mutant",)
# ax.scatter(y=Srt.index, x=df2.loc[Srt.index], s=900*np.power(sz2,2), c='blue', alpha=0.7,
#            label="Wild")


# ax.vlines(x=0, ymin=0, ymax=len(df)-1, color='black', alpha=0.7, linewidth=2, linestyles='dotted')
# plt.legend(fontsize=20,
#            facecolor='White', framealpha=1,frameon=True)

# ax.set_title('Mean of Distribution - '+F1+'/'+F2, fontdict={'size':22})
# ax.set_xlim(-1.5, 1.5)
# plt.savefig("Plots/"+F1+"_"+F2+"_Means.png",dpi=100)
# plt.show()

# df=dfb
# '''

# ### PairWise Plots



# title=F1+' Open Chromatin'
# plt.figure(figsize=(6, 5))    
# sns.pairplot(df[Open_Chromatin],corner=True,plot_kws={"s": 3},diag_kind="kde")
# plt.title(title)
# plt.savefig("Plots/"+F1+"_PW_Open.png",dpi=100)
# plt.show()

# title=F2+' Open Chromatin'
# plt.figure(figsize=(6, 5))    
# sns.pairplot(dfmut[Open_Chromatin],corner=True,plot_kws={"s": 3},diag_kind="kde")
# plt.title(title)
# plt.savefig("Plots/"+F2+"_PW_Open.png",dpi=100)
# plt.show()

# title=F1+' Closed Chromatin'
# plt.figure(figsize=(6, 5))    
# sns.pairplot(df[Closed_Chromatin],corner=True,plot_kws={"s": 3},diag_kind="kde")
# plt.title(title)
# plt.savefig("Plots/"+F1+"_PW_Closed.png",dpi=100)
# plt.show()

# CompVars=[
#     'H4K16ac',
#     'H3K4me1',
#     'H3K4me3',
#     'H2Aub',
#     'H3K27me3',
#     'Type']

# size=min(len(dfmut),len(df))
# EqSize=aaa[aaa['Type']=="Mutant"].iloc[0:size]
# EqSize=EqSize.append(aaa[aaa['Type']=="Wild"].iloc[0:size])




# title=F1+'/'+F2
# plt.figure(figsize=(6, 5))    
# sns.pairplot(EqSize[CompVars].sort_values("Type",ascending=False),
#              corner=True,hue="Type",plot_kws={"s": 3},
#         )
# plt.title(title)
# plt.savefig("Plots/"+F1+"_"+F2+"_PW.png",dpi=100)
# plt.show()



# Open_Chromatin.append('Type')
# title=F1+'/'+F2+" Open"
# plt.figure(figsize=(6, 5))    
# sns.pairplot(EqSize[Open_Chromatin].sort_values("Type",ascending=False),
#              corner=True,hue="Type",plot_kws={"s": 3},
#         )
# plt.title(title)
# plt.savefig("Plots/"+F1+"_"+F2+"_Open_PW.png",dpi=100)
# plt.show()

# '''
# ####

# Names=['H2Aub',
# # 'H2B',
# # 'H3',
# # 'H3.3',
#  'H3K27M',
#  'H3K27me3',
#  'H3K36me2',
#  'H3K36me3',
#  'H3K4me1',
#  'H3K4me3',
#  'H3K64ac',
#  'H3K9ac',
#  'H3K9me3',
# # 'H4',
#  'H4K16ac',
#  'cleaved H3',
#  #'H2A',
#  'pH3[S28]']

# dfb=df
# sns.set_style({'legend.frameon':True})
# df=np.mean(aaa[aaa['Type']=="Mutant"][Names]).sort_values()
# s1=np.full((len(df),4),0,dtype=np.float64)
# sz1=np.full(len(df),0,dtype=np.float64)
# df2=np.mean(aaa[aaa['Type']=="WT"][Names]).sort_values()
# s2=np.full((len(df2),4),0,dtype=np.float64)
# sz2=np.full(len(df2),0,dtype=np.float64)

# Srt=df-df2
# Srt=Srt.sort_values(ascending=False)


# for i in range(0,len(df)):
#     print(str(i)+" "+df.index[i])
#     q=np.std(aaa[aaa['Type']=="Mutant"][df.index[i]])*1/1.25
#     print(q)
#     s1[i]=[float(q),0,0,q]
#     sz1[i]=q
#     print(str(i)+" "+df2.index[i])
#     q=np.std(aaa[aaa['Type']=="WT"][df2.index[i]])*1./1.25
#     print(q)
#     s2[i]=[0,0,float(q),q]
#     sz2[i]=q


# diffs=(df-df2).sort_values(ascending=False)    
 
# colors = ['blue' if x < 0 else 'red' for x in diffs]



# # Draw plot
# fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
# plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=1, linewidth=5)

# # Decorations
# plt.gca().set(ylabel='', xlabel='')
# #plt.yticks(df.index, df.cars, fontsize=12)
# plt.title('Mean of Distribution - '+F1+'/'+F2, fontdict={'size':22})
# plt.grid(linestyle='--', alpha=0.5)
# plt.show()





### Graphics Setup
large = 22; med = 16; small = 14
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
###
matrix_df = pps.matrix(df[NamesUMAP])[['x', 'y', 'ppscore']].pivot(columns='x', index='y', 
                                                                                    values='ppscore')
sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="jet", linewidths=0.5, annot=True)
    
