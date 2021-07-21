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
    return np.std(od['H3'])+np.std(od['H3.3'])+np.std(od['H4'])
                  #(pow(od['H3']-avMarkers,2)+pow(od['H3.3']-avMarkers,2)+pow(od['H4']-avMarkers,2))





class RunData(dt.DataSet):
    """
    Execution data
    """
    #
    _bgCool = dt.BeginGroup('Dataset')
    case  = di.StringItem("Case name", default='C14')
    _egCool = dt.EndGroup('Dataset')
    #
#    _bgClu1 = dt.BeginGroup("Cluster 1")
#    u  = di.FloatItem("Velocity [m/s]", min=0, default=20.0)
#    u1  = di.StringItem("Labels 1", default='H2-18O v1')
#    _egClu1 = dt.BeginGroup("Cluster 1")
    _bgJet = dt.BeginGroup("Clusters")
    u1  = di.StringItem("Cluster 1", default='')
    u2  = di.StringItem("Cluster 2", default='')
    u3  = di.StringItem("Cluster 3", default='')
    _egJet = dt.EndGroup("Clusters")



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


def draw_umap(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''
              ,cc=0):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(ddf);
    plt.figure(figsize=(6, 5))
    if n_components == 2:
        plt.scatter(u[:,0], u[:,1], c=cc,s=3,cmap=plt.cm.seismic)
        plt.clim(-5,5)
        plt.colorbar()
    plt.title(title, fontsize=18)

#df=pd.read_csv("control.csv")
#dfmut=pd.read_csv("mutant.csv")
dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets/C13C14/"
df=pd.read_csv(dir+"C13.csv")
dfmut=pd.read_csv(dir+"C14.csv")



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
 'pHistone H2A.X [Ser139]',
 #'H2A',
 'pHistone H3 [S28]']



df=df[Names]
dfmut=dfmut[Names]

plt.figure(figsize=(6, 5))


df=df[df['H4']>5]
df=df[df['H3.3']>5]

dfmut=dfmut[dfmut['H4']>5]
dfmut=dfmut[dfmut['H3.3']>5]






Var1='H3K27M'
Var2='H3K36me2'


scFac=5
df=np.arcsinh(df/scFac)
dfmut=np.arcsinh(dfmut/scFac)



aaaa=df.append(dfmut)
m=np.mean(aaaa)
s=np.std(aaaa)
ldf=len(df)



EqualWeights=0
if EqualWeights==1:
    ddf=dfmut
    ddf=(ddf-m)/s
    avConstMarkers=np.sum(ddf[['H3.3','H4']]/2,axis=1)
    ddf=ddf.subtract(avConstMarkers,axis=0)
    dfmut=ddf

    ddf=df
    ddf=(ddf-m)/s
    avConstMarkers=np.sum(ddf[['H3.3','H4']]/2,axis=1)
    ddf=ddf.subtract(avConstMarkers,axis=0)
    df=ddf
if EqualWeights==0:
    params = Parameters()
    params.add('alpha', value=0.5, min=0)
    params.add('beta', value=0.5, min=0)
    params.add('gamma', value=0.1, min=0)
    aaaa=(aaaa-m)/s
#    out = minimize(residual, params, args=(aaaa, aaaa),method='cg')
#    alpha=out.params['alpha'].value
#    beta=out.params['beta'].value
#    gam=out.params['gamma'].value
    
    ddf=dfmut
    ddf=(ddf-m)/s
    out = minimize(residual, params, args=(ddf, ddf),method='cg')
    alpha=out.params['alpha'].value
    beta=out.params['beta'].value
    gam=out.params['gamma'].value
    avMarkers=ddf['H3.3']*alpha+ddf['H4']*beta+ddf['H3']*gam
    ddf=ddf.subtract(avMarkers,axis=0)
    dfmut=ddf

    ddf=df
    ddf=(ddf-m)/s
    out = minimize(residual, params, args=(ddf, ddf),method='cg')
    alpha=out.params['alpha'].value
    beta=out.params['beta'].value
    gam=out.params['gamma'].value
    avMarkers=ddf['H3.3']*alpha+ddf['H4']*beta+ddf['H3']*gam
    ddf=ddf.subtract(avMarkers,axis=0)
    df=ddf




Var='H3K27M'

#hist, bins = np.histogram(df[(df['H3K27ac']<4000)]['H3K27ac'], bins=150)
hist, bins = np.histogram(df[Var], bins=50)
#hist=hist/sum(hist)
err=np.sqrt(hist)

center = (bins[:-1] + bins[1:]) / 2

#histm, bins = np.histogram(dfmut[(dfmut['H3K27ac']<4000)]['H3K27ac'], bins=150)
histm, bins = np.histogram(dfmut[Var], bins=50)
#histm=histm/sum(histm)
errm=np.sqrt(hist)
centerm = (bins[:-1] + bins[1:]) / 2
plt.figure(figsize=(6, 5))
plt.plot(center,hist,'r.')
plt.plot(centerm,histm,'b.')
plt.title(Var)

Var='H3K27ac'

#hist, bins = np.histogram(df[(df['H3K27ac']<4000)]['H3K27ac'], bins=150)
hist, bins = np.histogram(df[Var], bins=50)
#hist=hist/sum(hist)
err=np.sqrt(hist)

center = (bins[:-1] + bins[1:]) / 2

#histm, bins = np.histogram(dfmut[(dfmut['H3K27ac']<4000)]['H3K27ac'], bins=150)
histm, bins = np.histogram(dfmut[Var], bins=50)
#histm=histm/sum(histm)
errm=np.sqrt(hist)
centerm = (bins[:-1] + bins[1:]) / 2
plt.figure(figsize=(6, 5))
plt.plot(center,hist,'r.')
plt.plot(centerm,histm,'b.')
plt.title(Var)


plt.figure(figsize=(6, 5))
plt.matshow(df.corr())
plt.colorbar()
plt.title('C14')
plt.clim(-1,1)

plt.figure(figsize=(6, 5))
plt.matshow(dfmut.corr())
plt.colorbar()
plt.title('C13')
plt.clim(-1,1)

'''
plt.figure(figsize=(6, 5))
diff=(dfmut.drop(columns='H3K27M').corr(method='spearman')-df.corr(method='spearman'))
plt.matshow(diff)
plt.colorbar()
plt.title('Diff Correleations Spearman')
plt.clim(-.5,.5)

plt.figure(figsize=(6, 5))
#diff=np.corrcoef(np.transpose(dfmut.drop(columns='H3K27M')))-np.corrcoef(np.transpose(df))
plt.matshow(diff)
plt.colorbar()
plt.title('Diff Correleations Pearson')
plt.clim(-.5,.5)
'''

Var1='H3K9me3'
Var2='H3K27ac'

plt.figure(figsize=(6, 5))
#plt.scatter(df[Var1],df[Var2],s=2,c='r')
plt.scatter(dfmut[Var1],dfmut[Var2],s=2,c='b')
plt.title(Var1+' vs. '+Var2 + ' Mutant')





df1=df[[Var1,Var2]]
dist = bp.EmpiricalDistribution(df1)
dist.sample(reset_index=False)

df1m=dfmut[[Var1,Var2]]
distm = bp.EmpiricalDistribution(df1m)
distm.sample(reset_index=False)

se,theta_star=bp.standard_error(dist, statistic,return_samples=True,B=1000)
print('Control Correlation ',Var1,'/',Var2,' ',df.corr().loc[Var1,Var2],' ',
      'SE',se)

se_m,theta_star_m=bp.standard_error(distm, statistic,return_samples=True,B=1000)
print('Mutant Correlation ',Var1,'/',Var2,' ',dfmut.corr().loc[Var1,Var2],' ',
      'SE',se_m)





plt.figure(figsize=(6, 5))
hist, bins = np.histogram(theta_star, bins=50)
center = (bins[:-1] + bins[1:]) / 2
plt.plot(center,hist,'r.')

histm, binsm = np.histogram(theta_star_m, bins=50)
centerm = (binsm[:-1] + binsm[1:]) / 2
plt.plot(centerm,histm,'b.')
plt.title('Bootstrapped Corrleations '+Var1+'/'+Var2)
plt.xlim([-1,1])
plt.legend(['Control','Mutant'])
#ci_low, ci_high,theta_star = bp.bcanon_interval(dist, statistic, df,
#                                                B=2,return_samples=True)
#print('Correlation ',Var1,'/',Var2,' ',se)#ci_low,' ',ci_high)

for NN in Names:
    ks,p=stats.ks_2samp(df.sample(5000)[NN],dfmut.sample(5000)[NN])
    col1 = ws.DescrStatsW(df.sample(5000)[NN])
    col2 = ws.DescrStatsW(dfmut.sample(5000)[NN])
    cm_obj = ws.CompareMeans(col1, col2)
    zstat, z_pval = cm_obj.ztest_ind(usevar='unequal')
    print(NN,' KS p-Value: ',p.round(4),' Z p-Val: ',z_pval.round(4))




####
NNN=Names
Names=['H2AK119ub',
 'H2B',
 'H3',
 'H3.3',
 'H3K27M',
 'H3K27me3',
 'H3K36me2',
 'H3K36me3',
 'H3K4me1',
 'H3K4me3',
 'H3K64ac',
 'H3K27ac',
 'H3K9ac',
 'H3K9me3',
 'H4',
 'H4K16Ac',
 'cleaved H3',
 'pHistone H2A.X [Ser139]',
 #'H2A',
 'pHistone H3 [S28]']

ddf=dfmut[Names].sample(frac=1)

ar=ddf#np.asarray(ddf)
tsne = TSNE(n_components=2, random_state=0)

#mask = np.random.choice([False, True], len(ar), p=[0.8, 0.2])

X_2d = tsne.fit_transform(ar)#[mask,:])


for NN in Names:
    Var=NN
    TSNEVar=NN
    cc=ddf[TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:,0],X_2d[:,1],s=2,
                c=cc, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(-3.5,3.5)
    plt.title(TSNEVar)



X=X_2d
km = KMeans(n_clusters=10,tol=1e-5)
km.fit(X)
km.predict(X)
plt.figure(figsize=(10, 10))
labels = km.labels_#Plotting
u_labels = np.unique(labels)
for i in u_labels:
    plt.scatter(X[labels == i , 0] , X[labels == i , 1] , label = i,s=100)
plt.legend(fontsize=15, title_fontsize='40')
#plot.legend(loc=2)
#plt.setp(ax.get_legend().get_texts(), fontsize='22')

plt.show()


aaa=ddf.assign(lbl=labels)
aaa=aaa.assign(tsne0=X[:,0])
aaa=aaa.assign(tsne1=X[:,1])
dfmut1=dfmut
dfmut=aaa
Names=NNN
####UMAP Section
'''
fit = umap.UMAP(n_neighbors=15)
%time u = fit.fit_transform(ddf)

X=u
km = KMeans(n_clusters=10,tol=1e-5)
km.fit(X)
km.predict(X)
plt.figure(figsize=(10, 10))
labels = km.labels_#Plotting
u_labels = np.unique(labels)
for i in u_labels:
    plt.scatter(X[labels == i , 0] , X[labels == i , 1] , label = i,s=100)
plt.legend(fontsize=15, title_fontsize='40')

plt.show()


for NN in Names:
    Var=NN
    TSNEVar=NN
    cc=ddf[TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(u[:,0],u[:,1],s=2,
                c=cc, cmap=plt.cm.seismic)
    plt.colorbar()
    plt.clim(-5,5)
    plt.title("UMAP "+TSNEVar)

###

'''

lab=dbscan_plot(X_2d,eps=0.181,min_samples=115)
aaaa=aaa.assign(clust=lab)


sns.set_style({'legend.frameon':True})
cl1=np.mean(aaaa[aaaa['clust']==1][Names]).sort_values()
cl2=np.mean(aaaa[aaaa['clust']==0][Names]).sort_values()
cl3=np.mean(aaaa[aaaa['clust']==2][Names]).sort_values()


fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.hlines(y=cl1.index, xmin=-5, xmax=5, color='gray', alpha=0.7, 
          linewidth=1, linestyles='dashdot')
ax.scatter(y=cl1.index, x=cl1, s=200, color='firebrick', alpha=0.7,
           label="Cluster 1",)
ax.scatter(y=cl2.index, x=cl2, s=200, color='blue', alpha=0.7,
           label="Cluster 2")
#ax.scatter(y=cl3.index, x=cl3, s=200, color='Magenta', alpha=0.7,
#           label="Cluster 3")
ax.vlines(x=0, ymin=0, ymax=len(cl1)-1, color='black', alpha=0.7, linewidth=2, linestyles='dotted')
plt.legend(fontsize=20,
           facecolor='White', framealpha=1,frameon=True)
ax.set_title('Mean of Distribution', fontdict={'size':22})
ax.set_xlim(-2.5, 2.5)

plt.show()

'''
master = Tk()
master.withdraw() #hiding tkinter window
_app = guidata.qapplication() 
param=RunData()
param.edit()


q=param.u1.split(',')
j=0; 
for i in q:
    q[j]=np.int(i);
    j=j+1;
Clus1=q

q=param.u2.split(',')
j=0; 
for i in q:
    q[j]=np.int(i);
    j=j+1;
Clus2=q

q=param.u3.split(',')
j=0; 
for i in q:
    q[j]=np.int(i);
    j=j+1;
Clus3=q

df_ = pd.DataFrame(columns=list(dfmut.columns))



for i in Clus1:
    mask=(labels==i)
    ddd=dfmut[mask].assign(clus=1)
    df_=df_.append(ddd)
    
    
for i in Clus2:
    mask=(labels==i)
    ddd=dfmut[mask].assign(clus=2)
    df_=df_.append(ddd)
    
for i in Clus3:
    mask=(labels==i)
    ddd=dfmut[mask].assign(clus=3)
    df_=df_.append(ddd)
    
for NN in Names:
    BoxVar=NN
    plt.figure(figsize=(6, 5))    
    ax = sns.boxplot(x="clus", y=NN, data=df_)
    plt.title(NN)
'''    

#CutVar='H3K27M'

#Top=dfmut[dfmut[CutVar]> dfmut[CutVar].quantile(0.75)]
#Mid=dfmut[((dfmut[CutVar] > dfmut[CutVar].quantile(0.25)) & 
#           (dfmut[CutVar] < dfmut[CutVar].quantile(0.75)))]
#Bottom=dfmut[dfmut[CutVar] < dfmut[CutVar].quantile(0.25)]





'''
#Generate Historgrams for all

for NN in Names:
    plt.figure(figsize=(6, 5))
    Var=NN
    hist, bins = np.histogram(Top[Var], bins=50)
    center = (bins[:-1] + bins[1:]) / 2
    hist_b, bins = np.histogram(Bottom[Var], bins=50)
    center_b = (bins[:-1] + bins[1:]) / 2
    hist_m, bins = np.histogram(Mid[Var], bins=50)
    center_m = (bins[:-1] + bins[1:]) / 2
    plt.plot(center,hist,'r.',center_m,hist_m,'m.',center_b,hist_b,'b.')
    hist, bins = np.histogram(Bottom[Var], bins=50)
    plt.title(Var+' Cut on '+CutVar)
    plt.legend(['High','Mid','Low'])
    plt.xlim([-5,5])
'''

#df_.to_csv(param.case+'.csv')