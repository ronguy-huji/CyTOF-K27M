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
import matplotlib 
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pacmap

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
                 markeredgecolor='k', markersize=6)
        
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=3)
    
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

def f(): raise Exception("Found exit()")

#df=pd.read_csv("control.csv")
#dfmut=pd.read_csv("mutant.csv")
dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets/C06C08/"
C06=pd.read_csv(dir+"C06.csv")
C08=pd.read_csv(dir+"C08_JustInduced.csv")


renamelist={'H2AK119ub' : 'H2Aub','pHistone H3 [S28]':'pH3[S28]','pHistone H2A.X [Ser139]':'yH2A.X',
            'H4K16Ac':'H4K16ac'}
C06.rename(columns=renamelist,inplace=True)
C08.rename(columns=renamelist,inplace=True)


#C11=C11[(C11 != 0).all(1)]
#C12=C12[(C12 != 0).all(1)]
lC06=len(C06)
print("C06")
for col in C06.columns:
    lG=  np.count_nonzero(C06[col]>1)  
    print("%-30s %d %5.3f" % (col,lG,lG/lC06*100.)) 
print(" ")
lC08=len(C08)
print("C08")
for col in C08.columns:
    lG=  np.count_nonzero(C08[col]>1)  
    print("%-30s %d %5.3f" % (col,lG,lG/lC08*100.)) 





NamesAll=['H3',
 'H3K36me3',
# 'H2B',
 'H3K4me3',
# 'yH2A.X',
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
 'pH3[S28]']

EpiGenList=['H3',
 'H3K36me3',
# 'H2B',
 'H3K4me3',
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
 'pH3[S28]']

DispList=[
# 'H3',
 'H3K36me3',
# 'H2B',
 'H3K4me3',
 'H3K36me2',
 'H4K16ac',
 'H2Aub',
 'H3K4me1',
# 'H3.3',
 'H3K64ac',
# 'H4',
 'H3K27ac',
 'cleaved H3',
 'H3K9ac',
 'H3K27me3',
 'H3K27M',
 'H3K9me3',
 'pH3[S28]']


Names_UMAP=['H3',
 'H3K36me3',
# 'H2B',
 'H3K4me3',
# 'pHistone H2A.X [Ser139]',
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
# 'H3K27M',
 'H3K9me3',
# 'pH3[S28]'
 ]



plt.figure(figsize=(6, 5))

GateColumns=['H3.3','H4','H3']

            
C06=C06[(C06[GateColumns]>5).all(axis=1)]
C08=C08[(C08[GateColumns]>5).all(axis=1)]


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
 'H3K27ac',
 'H3K9ac',
 'H3K27me3',
 'H3K9me3',
]


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

C06=C06[(C06[GateColumns]>0).all(axis=1)]
C08=C08[(C08[GateColumns]>0).all(axis=1)]

C06mask27M=C06['H3K27M']<1
C08mask27M=C08['H3K27M']<1
#C06maskpH3=C06['pHistone H3 [S28]']==0
#C08maskpH3=C08['pHistone H3 [S28]']==0

C06mask=C06<2
C08mask=C08<2


dfmask=C06mask.append(C08mask)


C08Bck=C08.copy()

scFac=5
C06=np.arcsinh(C06/scFac)
C08=np.arcsinh(C08/scFac)


aaaa=C06.append(C08)
m=np.mean(aaaa)
s=np.std(aaaa)



EqualWeights=0
if EqualWeights==1:
    ddf=C06
    ddf=(ddf-m)/s
    avConstMarkers=np.sum(ddf[['H3.3','H4']]/2,axis=1)
    ddf=ddf.subtract(avConstMarkers,axis=0)
    C06=ddf

    ddf=C08
    ddf=(ddf-m)/s
    avConstMarkers=np.sum(ddf[['H3.3','H4']]/2,axis=1)
    ddf=ddf.subtract(avConstMarkers,axis=0)
    C08=ddf

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
    
    ddf=C06
    ddf=(ddf-m)/s
    out = minimize(residual, params, args=(ddf, ddf),method='cg')
    alpha=out.params['alpha'].value
    beta=out.params['beta'].value
    gam=out.params['gamma'].value
    avMarkers=ddf['H3.3']*alpha+ddf['H4']*beta+ddf['H3']*gam
    ddf=ddf.subtract(avMarkers,axis=0)
    C06=ddf
    
    ddf=C08
    ddf=(ddf-m)/s
    out = minimize(residual, params, args=(ddf, ddf),method='cg')
    alpha=out.params['alpha'].value
    beta=out.params['beta'].value
    gam=out.params['gamma'].value
    avMarkers=ddf['H3.3']*alpha+ddf['H4']*beta+ddf['H3']*gam
    ddf=ddf.subtract(avMarkers,axis=0)
    C08=ddf    

#    ddf[NMS]=C18_Bck[NMS]
#    C18=ddf






#C06.loc[C06mask27M,'H3K27M']=-100
#C08.loc[C08mask27M,'H3K27M']=-100
#C06.loc[C06maskpH3,'pHistone H3 [S28]']=-100
#C08.loc[C08maskpH3,'pHistone H3 [S28]']=-100

C06=C06.assign(type='WT')
C08=C08.assign(type='Mutant')


CAll=C06.append(C08)

C06Bck=C06.copy()
C08Bck=C08.copy()

#CAll=C08
#dfmask=C08mask
#C08Bck=C08.copy()
### tSNE C18
ar=CAll[Names_UMAP]#np.asarray(ddf)


import time
tic = time.perf_counter()
X_2d=draw_umap(CAll[Names_UMAP],cc=CAll['H3K27M'],min_dist=0.05)
toc = time.perf_counter() 
print(f"Time {toc - tic:0.4f} seconds")





from scipy.spatial import ConvexHull

def area(p):
    return 0.5 * abs(sum(x0*y1 - x1*y0
                         for ((x0, y0), (x1, y1)) in segments(p)))

def segments(p):
    return zip(p, p[1:] + [p[0]])


def encircle(x,y, ax=None, **kw):
    if not ax: ax=plt.gca()
    p = np.c_[x,y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices,:], **kw)
    ax.add_patch(poly)
    return(hull)

plt.show()

CAll=CAll.assign(umap0=X_2d[:,0])
CAll=CAll.assign(umap1=X_2d[:,1])
lab=dbscan_plot(X_2d,eps=0.13,min_samples=50)

plt.figure(figsize=(6, 5))    
mask=lab==0
plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],s=2)
hhh=encircle(X_2d[mask][:,0],X_2d[mask][:,1],ec="k", fc="gold", alpha=0.2)
print(f"WT Density: {mask.sum()/hhh.area}")
mask=lab==1
plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],s=2)
hhh=encircle(X_2d[mask][:,0],X_2d[mask][:,1],ec="k", fc="red", alpha=0.2)
print(f"Mutant Density: {mask.sum()/hhh.area}")

# import time
# tic = time.perf_counter()
# PCMap=pacmap.pacmap.PaCMAP(n_dims=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
# XX=CAll[Names_UMAP].values
# X_2d=PCMap.fit_transform(XX,init="pca")
# toc = time.perf_counter() 
# print(f"Time {toc - tic:0.4f} seconds")

# f()

# for NN in NamesAll:
#     Var=NN
#     TSNEVar=NN
#     cc=CAllRed[TSNEVar]#[mask]
#     plt.figure(figsize=(6, 5))
#     plt.scatter(CAllRed["umap0"],CAllRed["umap1"],s=1,
#                 c=cc, cmap=plt.cm.jet)
#     plt.colorbar()
#     plt.clim(-3.5,3.5)
#     plt.clim(cc.quantile(0.01),cc.quantile(0.99))
#     cmap = matplotlib.cm.get_cmap('jet')
#     mask=dfmask[TSNEVar]==True
#     rgba = cmap(-10)
#     plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],s=1,
#                 color=rgba) 
#     plt.title(TSNEVar+" C06/C08")
#     plt.savefig("/Users/ronguy/Dropbox/Work/CyTOF/Plots/UMAP_"+TSNEVar+"_C06C08.pdf",
#                 dpi=100,bbox_inches = 'tight')
#     plt.show()


for NN in NamesAll:
    Var=NN
    TSNEVar=NN
    cc=CAll[TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:,0],X_2d[:,1],s=2,
                c=cc, cmap=plt.cm.seismic)
    plt.colorbar()
    plt.clim(-3.5,3.5)
    plt.clim(cc.quantile(0.01),cc.quantile(0.99))
    cmap = matplotlib.cm.get_cmap('jet')
    mask=dfmask[TSNEVar]==True
    rgba = cmap(-10)
    plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],s=2,
                color=rgba) 
    plt.title(TSNEVar+" C06/C08")
    plt.savefig("/Users/ronguy/Dropbox/Work/CyTOF/Plots/UMAP_"+TSNEVar+"_C06C08.png",
                dpi=100,bbox_inches = 'tight')
    plt.show()



X=X_2d
'''
km = KMeans(n_clusters=10,tol=1e-5)
km.fit(X)
km.predict(X)
plt.figure(figsize=(10, 10))
labels = km.labels_#Plotting
u_labels = np.unique(labels)
for i in u_labels:
    plt.scatter(X[labels == i , 0] , X[labels == i , 1] , label = i,s=100)
plt.legend(fontsize=15, title_fontsize='40')
'''

CAll=CAll.assign(umap0=X[:,0])
CAll=CAll.assign(umap1=X[:,1])
lab=dbscan_plot(X_2d,eps=0.13,min_samples=50)

#C18=C18.assign(clust=lab)






sns.set_style({'legend.frameon':True})
#CAll=CAll[CAll['clust'].isin([0,1])]
d0=C06.copy()#CAll[CAll['type']=='WT' ]
d1=C08.copy()#CAll[CAll['type']=='Mutant' ]

#d0[C06mask]=float("nan")
#d1[C08mask]=float("nan")


#d0=d0.drop(columns=['H3','H3.3','H4'])
#d1=d1.drop(columns=['H3','H3.3','H4'])


dd0=d0[DispList].mean().sort_values(ascending=False)
dd1=d1[DispList].mean().sort_values()


#dd0=dd0.drop(labels=['H3','H3.3','H4'])
#dd1=dd1.drop(labels=['H3','H3.3','H4'])


sz0=np.full(len(dd0),0,dtype=np.float64)
sz1=np.full(len(dd1),0,dtype=np.float64)
sz2=np.full(len(dd1),0,dtype=np.float64)
sz3=np.full(len(dd1),0,dtype=np.float64)



for i in range(0,len(dd0)):
#    print(str(i)+" "+dd0.index[i])
    q=np.std(d0[dd0.index[i]])*1/1.25
    sz0[i]=q
#    print(q)
 
    q=np.std(d1[dd1.index[i]])*1/1.25
    sz1[i]=q
    
 

diffs=(dd1-dd0).sort_values(ascending=False)    
 
colors = ['blue' if x < 0 else 'red' for x in diffs]



# Draw plot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=1, linewidth=5)

# Decorations
plt.gca().set(ylabel='', xlabel='')
plt.xticks(fontsize=30 ) 
plt.yticks(fontsize=30 ) 
#plt.yticks(df.index, df.cars, fontsize=12)
plt.title('C08 C06 Diffs', fontdict={'size':20})
plt.grid(linestyle='--', alpha=0.5)
plt.show()

### Box Plots 
CAll=C06.assign(Type='WT')
CAll=CAll.append(C08.assign(Type='Mutant'))
for NN in DispList:
     BoxVar=NN
     plt.figure(figsize=(6, 5))    
     ax = sns.boxplot(x="Type", y=NN, data=CAll,showfliers=False,palette=['blue','red'])
     #plt.title(NN+" C08 Cell Cycle")
     plt.show()



### Correlations


DispList2=[
# 'H3',
 'H3K36me3',
# 'H2B',
 'H3K4me3',
 'H3K36me2',
 'H4K16ac',
 'H2Aub',
 'H3K4me1',
# 'H3.3',
 'H3K64ac',
# 'H4',
 'H3K27ac',
 'cleaved H3',
 'H3K9ac',
 'H3K27me3',
# 'H3K27M',
 'H3K9me3',
 'pH3[S28]']

d0[C06mask]=float("nan")
d1[C08mask]=float("nan")




sns.set(font_scale=1.2)

print("Cluster map C06")
plt.figure(figsize=(10,10))
matrix=d0[DispList2].corr()
g=sns.clustermap(matrix, annot=True,
        cmap=plt.cm.jet,vmin=-.2,vmax=1,linewidths=.1); 
plt.xticks(rotation=0); 
plt.yticks(rotation=0); 
#plt.xticks(fontsize=20 ) 
#plt.yticks(fontsize=20 ) 
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.title('C06')
plt.show()

print("Cluster map C08")
plt.figure(figsize=(10,10))
matrix=d1[DispList2].corr()
g=sns.clustermap(matrix, annot=True,
        cmap=plt.cm.jet,vmin=-.2,vmax=1,linewidths=.1); 
plt.xticks(rotation=0); 
plt.yticks(rotation=0); 

plt.title('C08')
plt.show()

print("Cluster map C08-C06")
plt.figure(figsize=(10,10))
matrix=d1[DispList2].corr()-d0[DispList2].corr()
g=sns.clustermap(matrix, annot=True,
        cmap=plt.cm.jet,vmin=matrix.min().min(),vmax=matrix.max().max(),linewidths=.1); 
plt.xticks(rotation=0); 
plt.yticks(rotation=0); 

plt.title('C08-C06')
plt.show()




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


Open_Chromatin=[
     'H4K16ac','H3K4me1','H3K4me3','H3K27ac','H3K9ac','H3K64ac'
     ]
Closed_Chromatin=['H3K9me3', 'H3K27me3', 'H3K36me2', 'H2Aub', 'H3K9ac','H3K27ac']

print("Cluster map C06")
plt.figure(figsize=(10,10))
matrix=d0[Open_Chromatin].corr()
g=sns.clustermap(matrix, annot=True, annot_kws={"size":20},
        cmap=plt.cm.jet,vmin=-.2,vmax=1,linewidths=.1); 
plt.xticks(rotation=0); 
plt.yticks(rotation=0); 
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("/Users/ronguy/Dropbox/Work/CyTOF/Plots/C06_OpenCh.pdf")
plt.title('C06 Open Chromatin')
plt.show()


plt.figure(figsize=(10,10))
sns.pairplot(d0[Open_Chromatin],corner=True,
              diag_kind="kde",diag_kws=dict(fill=True, common_norm=False),
              )
plt.title('C06 Closed Chromatin',y=1.5)

plt.show()



print("Cluster map C06")
plt.figure(figsize=(10,10))
matrix=d0[Closed_Chromatin].corr()
g=sns.clustermap(matrix, annot=True, annot_kws={"size":20},
        cmap=plt.cm.jet,vmin=-.2,vmax=1,linewidths=.1); 
plt.xticks(rotation=0); 
plt.yticks(rotation=00); 
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("/Users/ronguy/Dropbox/Work/CyTOF/Plots/C06_ClosedCh.pdf")
plt.title('C06 Closed Chromatin')
plt.show()

plt.figure(figsize=(10,10))
sns.pairplot(d0[Closed_Chromatin],corner=True,
              diag_kind="kde",diag_kws=dict(fill=True, common_norm=False),
              )
plt.title('C06 Closed Chromatin',y=1.5)
plt.show()


######

print("Cluster map C08")
plt.figure(figsize=(10,10))
matrix=d1[Open_Chromatin].corr()
g=sns.clustermap(matrix, annot=True, annot_kws={"size":20},
        cmap=plt.cm.jet,vmin=-.2,vmax=1,linewidths=.1); 
plt.xticks(rotation=0); 
plt.yticks(rotation=0); 
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("/Users/ronguy/Dropbox/Work/CyTOF/Plots/C08_OpenCh.pdf")
plt.title('C08 Open Chromatin')
plt.show()




print("Cluster map C08")
plt.figure(figsize=(10,10))
matrix=d1[Closed_Chromatin].corr()
g=sns.clustermap(matrix, annot=True, annot_kws={"size":20},
        cmap=plt.cm.jet,vmin=-.2,vmax=1,linewidths=.1); 
plt.xticks(rotation=0); 
plt.yticks(rotation=00); 
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig("/Users/ronguy/Dropbox/Work/CyTOF/Plots/C08_ClosedCh.pdf")
plt.title('C08 Closed Chromatin')
plt.show()



#######



# import ppscore as pps
# matrix_df = pps.matrix(CAll)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', 
#                                                                                  values='ppscore')
# plt.figure(figsize=(20,20)) 
# sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="jet", linewidths=0.5, annot=True)
# plt.show()