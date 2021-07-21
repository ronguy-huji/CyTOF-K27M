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

#df=pd.read_csv("control.csv")
#dfmut=pd.read_csv("mutant.csv")
dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets/17321/"
C14=pd.read_csv(dir+"C16.csv")

#C11=C11[(C11 != 0).all(1)]
#C12=C12[(C12 != 0).all(1)]
# threshold=5
# lC13=len(C13)
# print("C13")
# for col in C13.columns:
#     lG=  np.count_nonzero(C13[col]>threshold)  
#     print("%-30s %d %5.3f" % (col,lG,lG/lC13*100.)) 
# print(" ")
# lC14=len(C14)
# print("C14")
# for col in C14.columns:
#     lG=  np.count_nonzero(C14[col]>threshold)  
#     print("%-30s %d %5.3f" % (col,lG,lG/lC14*100.)) 

NamesAll=['H3',
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



Names_UMAP=[
 'H3',
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
]



plt.figure(figsize=(6, 5))

GateColumns=['H3.3','H4','H3']

            
C14=C14[(C14[GateColumns]>5).all(axis=1)]

GateColumns=[
     'H3',
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
     'H3K9ac',
     'H3K27me3',
     'H3K9me3',]


C14=C14[(C14[GateColumns]>0).all(axis=1)]

threshold=2
lC14=len(C14)
print("C14")
for col in C14.columns:
    lG=  np.count_nonzero(C14[col]>threshold)  
    print("%-30s %d %5.3f" % (col,lG,lG/lC14*100.)) 
print(" ")


C14mask27M=C14['H3K27M']<threshold
C14maskpH3=C14['pH3[S28]']==0

C14mask=C14[NamesAll]<threshold
CAllmask=C14mask


IdUMask=C14['IdU']<150


scFac=5
C14=np.arcsinh(C14/scFac)

# for V in C14.columns:
#      C14[V] = stats.yeojohnson(C14[V])[0]

aaaa=C14
m=np.mean(aaaa)
s=np.std(aaaa)



EqualWeights=0
if EqualWeights==1:
    ddf=C13
    ddf=(ddf-m)/s
    avConstMarkers=np.sum(ddf[['H3.3','H4']]/2,axis=1)
    ddf=ddf.subtract(avConstMarkers,axis=0)
    C13=ddf

    ddf=C14
    ddf=(ddf-m)/s
    avConstMarkers=np.sum(ddf[['H3.3','H4']]/2,axis=1)
    ddf=ddf.subtract(avConstMarkers,axis=0)
    C14=ddf

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
    
    
    ddf=C14
    ddf=(ddf-m)/s
    out = minimize(residual, params, args=(ddf, ddf),method='cg')
    alpha=out.params['alpha'].value
    beta=out.params['beta'].value
    gam=out.params['gamma'].value
    avMarkers=ddf['H3.3']*alpha+ddf['H4']*beta+ddf['H3']*gam
    ddf=ddf.subtract(avMarkers,axis=0)
    C14=ddf    

#    ddf[NMS]=C18_Bck[NMS]
#    C18=ddf



#C14.loc[C14mask27M,'H3K27M']=-100
#C13.loc[C13maskpH3,'pHistone H3 [S28]']=-100
#C14.loc[C14maskpH3,'pHistone H3 [S28]']=-100


#C13[C13mask]=-10
#C14[C14mask]=-10

C14=C14.assign(type='Mutant')



C14=C14.assign(IdUHigh=~IdUMask)

CAll=C14
#CAllmask=CAllmask.assign(type=False)
#CAll[CAllmask]=-10
### tSNE C18
ar=CAll[Names_UMAP]#np.asarray(ddf)

X_2d=draw_umap(CAll[Names_UMAP],cc=CAll['H3K27M'],min_dist=0.01)



for NN in NamesAll:
    Var=NN
    TSNEVar=NN
    cc=CAll[TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:,0],X_2d[:,1],s=2,
                c=cc, cmap=plt.cm.jet)
    cmap = matplotlib.cm.get_cmap('jet')
    plt.colorbar()
    plt.clim(-3.5,3.5)
    mask=CAllmask[TSNEVar]==True
    rgba = cmap(-10)
    plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],s=2,
                color=rgba) 
    plt.title(TSNEVar+" C14 Epigenetic")



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
lab=dbscan_plot(X_2d,eps=0.3,min_samples=50)

CAll=CAll.assign(clust=lab)
CAll=CAll.assign(IdUHigh=~IdUMask)
C14mask=C14mask.assign(clust=lab)




'''

plt.figure(figsize=(6, 5))
mask=CAll.clust==0
plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],s=2,
                color='darkorange',label='H3K27M High')
mask=CAll.clust==1
plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],s=2,
                color='dimgray',label='H3K27M Low')
plt.legend(markerscale=5,fontsize=15)
plt.title("C14")
plt.show()



sns.set_style({'legend.frameon':True})
#CAll=CAll[CAll['clust'].isin([0,1])]


#d0[C13mask]=float("nan")
#d1[C14mask]=float("nan")

d0=CAll[CAll['clust']==0]
d1=CAll[CAll['clust']==1]
#d0=d0.drop(columns=['H3','H3.3','H4'])
#d1=d1.drop(columns=['H3','H3.3','H4'])

#d0[C13mask.all(axis=1)]=-10
#d1[C14mask.all(axis=1)]=-10

dd0=d0[NamesAll].mean().sort_values(ascending=False)
dd1=d1[NamesAll].mean().sort_values()


dd0=dd0.drop(labels=['H3','H3.3','H4'])
dd1=dd1.drop(labels=['H3','H3.3','H4'])


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
    
 

diffs=(dd0-dd1).sort_values(ascending=False)    
 
colors = ['dimgray' if x < 0 else 'darkorange' for x in diffs]



# Draw plot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=1, linewidth=5)

# Decorations
plt.gca().set(ylabel='', xlabel='')
#plt.yticks(df.index, df.cars, fontsize=12)
plt.title('C14 Clusters 27M high - low', fontdict={'size':20})
plt.grid(linestyle='--', alpha=0.5)
plt.show()

####
CAll=CAll[CAll.clust.isin([0,1])]
for NN in NamesAll:
     BoxVar=NN
     plt.figure(figsize=(3, 5))    
     ax = sns.boxplot(x="clust", y=NN, data=CAll,showfliers=False,palette=['darkorange','dimgray'])
     ax.set_xticklabels(['H3K27M \nHigh','H3K27M \nLow'])
     #plt.title(NN+" C16 Cell Cycle")
     plt.show()
'''
