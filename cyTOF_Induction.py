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
import matplotlib.pyplot as plt

#import guidata
#import guidata.dataset.datatypes as dt
#import guidata.dataset.dataitems as di

import pandas as pd
import umap
import seaborn as sns
#from time import time
#from bootstrap_stat import datasets as d
#from bootstrap_stat import bootstrap_stat as bp
#from sklearn.preprocessing import StandardScaler
#from sklearn import preprocessing
#from sklearn import datasets
#from sklearn.neural_network import MLPRegressor
#from sklearn.linear_model import LinearRegression
#import statsmodels.stats.weightstats as ws
#from sklearn.cluster import KMeans

#from tkinter import *
#from tkinter.filedialog import asksaveasfilename


from lmfit import minimize, Parameters
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


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
        plt.scatter(u[:,0], u[:,1], c=cc,s=3,cmap=plt.cm.seismic)
        plt.clim(-5,5)
        plt.colorbar()
    plt.title(title, fontsize=18)
    return u;
    
    


def residual(params, x, data):
    alpha = params['alpha']
    beta = params['beta']
#    gam = params['gamma']
 

    avMarkers=x['H3.3']*alpha+x['H4']*beta#+x['H3']*gam
    od=x.subtract(avMarkers,axis=0)
    return np.std(od['H3.3'])+np.std(od['H4'])#+np.std(od['H3'])
                  #(pow(od['H3']-avMarkers,2)+pow(od['H3.3']-avMarkers,2)+pow(od['H4']-avMarkers,2))


dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets/experiment_351890_exported_events_spill_applied_events/"
df=pd.read_csv(dir+"C01.csv")
dfmut=pd.read_csv(dir+"C02.csv")
dfmut1=pd.read_csv(dir+"C03.csv")
dfmut2=pd.read_csv(dir+"C04.csv")
dfmut3=pd.read_csv(dir+"C05.csv")


Names=['Cleaved H3',
 'H3K27me3',
 'H3K9me3',
 'H3.3',
 'BMI-1',
 'c-Myc',
 'H3K27ac',
 'H3K36me2',
 'H4K16ac',
 'H4',
 'H3K27M']


Names_All=['Cleaved H3',
 'H3K27me3',
 'H3K9me3',
 'H3.3',
 'BMI-1',
 'c-Myc',
 'H3K27ac',
 'H3K36me2',
 'H4K16ac',
 'H4',
 'H3K27M']


Names2=['Cleaved H3',
 'H3K27me3',
 'H3K9me3',
 'H3.3',
 'H3K27ac',
 'H3K36me2',
 'H4K16ac',
 'H4',
 'H3K27M']

LargeScale=[-1.2,0.75]
SmallScale=[-0.2,0.15]
YScale=[
    SmallScale,
    LargeScale,
    SmallScale,
    LargeScale,
    SmallScale,
    SmallScale,
    LargeScale,
    LargeScale,
    LargeScale,
    LargeScale,
    LargeScale    
        ]







df_=df
df_=df_.append(dfmut)
df_=df_.append(dfmut1)
df_=df_.append(dfmut2)
df_=df_.append(dfmut3)

l0=len(df)
l1=len(dfmut)
l2=len(dfmut1)
l3=len(dfmut2)
l4=len(dfmut3)

df_mask=df<2




scFac=5
df_=np.arcsinh(df_/scFac)


m=np.mean(df_)
s=np.std(df_)

ldf=len(df_)

EqualWeights=1
if EqualWeights==1:
    df_=(df_-m)/s
    avConstMarkers=np.sum(df_[['H3.3','H4']]/2,axis=1)
    df_=df_.subtract(avConstMarkers,axis=0)
if EqualWeights==0:

    params = Parameters()
    params.add('alpha', value=0.5, min=0)
    params.add('beta', value=0.5, min=0)   
    aa=df
    aa=(aa-m)/s
    out = minimize(residual, params, args=(aa, aa),method='cg')
    alpha=out.params['alpha'].value
    beta=out.params['beta'].value
    avMarkers=df['H3.3']*alpha+df['H4']*beta
    aa=aa.subtract(avMarkers,axis=0)
    df=aa
    
    aa=dfmut
    aa=(aa-m)/s
    out = minimize(residual, params, args=(aa, aa),method='cg')
    alpha=out.params['alpha'].value
    beta=out.params['beta'].value
    avMarkers=df['H3.3']*alpha+df['H4']*beta
    aa=aa.subtract(avMarkers,axis=0)
    dfmut=aa
    
    aa=dfmut1
    aa=(aa-m)/s
    out = minimize(residual, params, args=(aa, aa),method='cg')
    alpha=out.params['alpha'].value
    beta=out.params['beta'].value
    avMarkers=df['H3.3']*alpha+df['H4']*beta
    aa=aa.subtract(avMarkers,axis=0)
    dfmut1=aa
    
    aa=dfmut2
    aa=(aa-m)/s
    out = minimize(residual, params, args=(aa, aa),method='cg')
    alpha=out.params['alpha'].value
    beta=out.params['beta'].value
    avMarkers=df['H3.3']*alpha+df['H4']*beta
    aa=aa.subtract(avMarkers,axis=0)
    dfmut2=aa
    
    aa=dfmut3
    aa=(aa-m)/s
    out = minimize(residual, params, args=(aa, aa),method='cg')
    alpha=out.params['alpha'].value
    beta=out.params['beta'].value
    avMarkers=df['H3.3']*alpha+df['H4']*beta
    aa=aa.subtract(avMarkers,axis=0)
    dfmut3=aa
    
    df_=df
    df_=df_.append(dfmut)
    df_=df_.append(dfmut1)
    df_=df_.append(dfmut2)
    df_=df_.append(dfmut3)




d0=df_[0:l0].assign(ind=0)
d1=df_[l0:l0+l1].assign(ind=1)
d2=df_[l0+l1:l0+l1+l2].assign(ind=2)
d3=df_[l0+l1+l2:l0+l1+l2+l3].assign(ind=3)
d4=df_[l0+l1+l2+l3:l0+l1+l2+l3+l4].assign(ind=4)

d0=d0.assign(t=0)
d1=d1.assign(t=8)
d2=d2.assign(t=16)
d3=d3.assign(t=48)
d4=d4.assign(t=96)

df_=d0
df_=df_.append(d1)
df_=df_.append(d2)
df_=df_.append(d3)
df_=df_.append(d4)
#df_=df_.append(ddd)
#ddd=dfmut2.assign(int=3)
#df_=df_.append(ddd)



for NN in Names:
    Var=NN
    plt.figure(figsize=(6, 5))
    plt.title(Var)
    colors=['r','b','g','m','k']
    legends=['WT','1','2','3','4']
    for i in [0,1,2,3,4]:    
        df=df_[df_['ind']==i]
        hist, bins = np.histogram(df[Var], bins=50)
        center = (bins[:-1] + bins[1:]) / 2    
        plt.plot(center,hist,colors[i])
        plt.legend(legends)
        
for NN in Names:
    BoxVar=NN
    plt.figure(figsize=(6, 5))    
    ax = sns.boxplot(x="ind", y=NN, data=df_,showfliers = False,)
    plt.title(NN)

'''
for NN in Names:
    BoxVar=NN
    plt.figure(figsize=(6, 5))   
    aaa=df_[[NN,'ind']].groupby('ind').mean()
    ax = plt.plot(aaa.iloc[:,0],'b.',markersize=30)
    ax = plt.plot(aaa.iloc[:,0],'b-')
    plt.title(NN)
    plt.xlabel('Ind')
    plt.ylabel('Mean('+NN+')')
'''

TimeVars=[
 'H3K27me3',
 'H3K27ac',
 'H3K36me2',
 'H4K16ac',
 'H3K27M']

colors = plt.cm.jet(np.linspace(0, 1, len(TimeVars)))
markers= ['o','*','p','X','d']
marker_style = dict(linestyle='-',alpha=1)
plt.figure(figsize=(10, 5))   
i=0
for NN in TimeVars:
    BoxVar=NN
    aaa=df_[[NN,'t']].groupby('t').mean()
#    ax = plt.plot(aaa.iloc[:,0],'.',markersize=20,color=colors[i])
    ax = plt.plot(aaa.iloc[:,0],marker=markers[i],color=colors[i],label=NN,
                  markersize=8, **marker_style)
    plt.xlabel('Time [h]')
    plt.ylabel('Mean')
    plt.legend(loc=8)
    i=i+1
plt.title('Variation Over Time')

TimeVars=['Cleaved H3',
          'H3K9me3',
          ]

colors = plt.cm.jet(np.linspace(0, 1, len(TimeVars)))
markers= ['o','*','p','X','d']
marker_style = dict(linestyle='-')
plt.figure(figsize=(10, 5))   
i=0
for NN in TimeVars:
    BoxVar=NN
    aaa=df_[[NN,'t']].groupby('t').mean()
#    ax = plt.plot(aaa.iloc[:,0],'.',markersize=20,color=colors[i])
    ax = plt.plot(aaa.iloc[:,0],marker=markers[i],color=colors[i],label=NN,
                  markersize=8, **marker_style)
    plt.xlabel('Time [h]')
    plt.ylabel('Mean')
    plt.legend(loc=8)
    i=i+1
    plt.title('Variation Over Time')

'''
for IVar in [0,1,2,3,4]:
    plt.figure(figsize=(6, 5))
    df=df_[df_['ind']==IVar]
    plt.scatter(df['H3K27M'],df['H3K9me3'],s=2)
    plt.title('H3K27M vs. H3K9me3 '+str(IVar))
    plt.xlim(-5,5)
    plt.ylim(-5,5)
'''
for IVar in [0,1,2,3,4]:
    cv=df_[df_['ind']==IVar][Names2].corr()
    cv.to_excel(str(IVar)+".xlsx")
###    plt.matshow(cv)
###    plt.colorbar()
###    plt.title(str(IVar))
###    plt.clim(-1,1)
    

#df_.to_csv(param.case+'.csv')


NamesTSNE=[
 'Cleaved H3',
 'H3K27me3',
 'H3K9me3',
 'H3.3',
 'BMI-1',
 'c-Myc',
 'H3K27ac',
 'H3K36me2',
 'H4K16ac',
 'H4'
 ]



aaa=df_
ddf=aaa.sample(frac=0.1)

ar=ddf#np.asarray(ddf)
tsne = TSNE(n_components=2, random_state=0)

#X_2d = tsne.fit_transform(ddf[NamesTSNE])
X_2d=draw_umap(ddf[NamesTSNE],cc=ddf['H3.3'])

'''
fig = plt.figure();
ax = fig.add_subplot(111, projection='3d'); 
mask=ddf['ind']==0
ax.scatter(X_2d[mask][:,0],X_2d[mask][:,1],X_2d[mask][:,2],color='blue',s=2);
mask=ddf['ind']==1
ax.scatter(X_2d[mask][:,0],X_2d[mask][:,1],X_2d[mask][:,2],color='green',s=2);
mask=ddf['ind']==2
ax.scatter(X_2d[mask][:,0],X_2d[mask][:,1],X_2d[mask][:,2],color='yellow',s=2);
mask=ddf['ind']==3
ax.scatter(X_2d[mask][:,0],X_2d[mask][:,1],X_2d[mask][:,2],color='orange',s=2);
mask=ddf['ind']==4
ax.scatter(X_2d[mask][:,0],X_2d[mask][:,1],X_2d[mask][:,2],color='red',s=2);
ax.view_init(30, azim=30)
plt.show();
fig = plt.figure();
ax = fig.add_subplot(111, projection='3d'); 
mask=ddf['ind']==0
ax.scatter(X_2d[mask][:,0],X_2d[mask][:,1],X_2d[mask][:,2],color='blue',s=2);
mask=ddf['ind']==1
ax.scatter(X_2d[mask][:,0],X_2d[mask][:,1],X_2d[mask][:,2],color='green',s=2);
mask=ddf['ind']==2
ax.scatter(X_2d[mask][:,0],X_2d[mask][:,1],X_2d[mask][:,2],color='yellow',s=2);
mask=ddf['ind']==3
ax.scatter(X_2d[mask][:,0],X_2d[mask][:,1],X_2d[mask][:,2],color='orange',s=2);
mask=ddf['ind']==4
ax.scatter(X_2d[mask][:,0],X_2d[mask][:,1],X_2d[mask][:,2],color='red',s=2);
ax.view_init(30, azim=180)
plt.show()
'''



### tSNE T0

mask=(ddf['ind']==0)
plt.figure(figsize=(6, 5)) 
plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],color='blue',s=2,label='T=0h');
plt.title('T=0h UMAP')
for NN in Names:
    Var=NN
    TSNEVar=NN
    cc=ddf[mask][TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],s=2,
                c=cc, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(-3.,3.)
    plt.title(TSNEVar+" w/o K27M T=0h")

### tSNE T1

mask=(ddf['ind']==1)
plt.figure(figsize=(6, 5)) 
plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],color='green',s=2,label='T=8h');
plt.title('T=8h UMAP')
for NN in Names:
    Var=NN
    TSNEVar=NN
    cc=ddf[mask][TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],s=2,
                c=cc, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(-3.,3.)
    plt.title(TSNEVar+" w/o K27M T=8h")

### tSNE T2

mask=(ddf['ind']==2)
plt.figure(figsize=(6, 5)) 
plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],color='yellow',s=2,label='T=16h');
plt.title('T=16h UMAP')
for NN in Names:
    Var=NN
    TSNEVar=NN
    cc=ddf[mask][TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],s=2,
                c=cc, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(-3.,3.)
    plt.title(TSNEVar+" w/o K27M T=16h")

### tSNE T3

mask=(ddf['ind']==3)
plt.figure(figsize=(6, 5)) 
plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],color='orange',s=2,label='T=48h');
plt.title('T=48h UMAP')
for NN in Names:
    Var=NN
    TSNEVar=NN
    cc=ddf[mask][TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],s=2,
                c=cc, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(-3.,3.)
    plt.title(TSNEVar+" w/o K27M T=48h")

### tSNE T4

mask=(ddf['ind']==4)
plt.figure(figsize=(6, 5)) 
plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],color='red',s=2,label='T=96h');
plt.title('T=96h UMAP') 
for NN in Names:
    Var=NN
    TSNEVar=NN
    cc=ddf[mask][TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],s=2,
                c=cc, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(-3.,3.)
    plt.title(TSNEVar+" w/o K27M T=96h")
    

labs=["0 hr","8 hr","16 hr","48 hr","96 hr"]
plt.figure(figsize=(6, 5))
cList=['blue','green','yellow','orange','red']
for i in [0,1,2,3,4]:
    mask=ddf['ind']==i
    plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],color=cList[i],s=2,label=labs[i],alpha=0.2)

for i in [0,1,2,3,4]:    
    mask=ddf['ind']==i    
    plt.scatter(np.mean(X_2d[mask][:,0]),np.mean(X_2d[mask][:,1]),s=150,alpha=1,
                color=cList[i],edgecolors='black')

X=np.full(5,0,dtype=np.float64)
Y=np.full(5,0,dtype=np.float64)

for i in [0,1,2,3,4]:    
    mask=ddf['ind']==i    
    X[i]=np.mean(X_2d[mask][:,0])
    Y[i]=np.mean(X_2d[mask][:,1])

dX=np.diff(X)
dY=np.diff(Y)

for i in range(0,4):
    plt.arrow(X[i], Y[i], dX[i], dY[i],length_includes_head=True,width=.1,color='black')

leg=plt.legend(markerscale=5)

for lh in leg.legendHandles:
    lh.set_alpha(1.0)
plt.savefig("/Users/ronguy/Dropbox/Work/CyTOF/Plots/Induction_Prog.pdf")
plt.show()   
#df_.to_csv("Test.csv")

Var1='H3K27M'

i=0
for NN in Names:
    plt.figure(figsize=(6, 5))
    aaa=df_[[Var1,NN,'ind']].groupby('ind').mean()
    plt.plot(aaa.iloc[:,0],aaa.iloc[:,1],'b.-');
    plt.scatter(aaa.iloc[:,0],aaa.iloc[:,1],s=50,c='b');
    plt.xlabel(Var1);
    plt.ylabel(NN)
    plt.ylim(YScale[i])
    i=i+1



NamesCorr1=['Cleaved H3',
 'H3K27me3',
 'H3K9me3',
 'H3K27ac',
 'H3K36me2',
 'H4K16ac',
]

NamesCorr2=['Cleaved H3',
 'H3K27me3',
 'H3K9me3',
 'H3K27ac',
 'H3K36me2',
 'H4K16ac',
 'H3K27M']


plt.figure(figsize=(10,10))
sns.clustermap(df_[df_['ind']==0][NamesCorr1].corr(),annot=True,
            cmap=plt.cm.jet,vmin=-.2,vmax=1,linewidths=.1); 
plt.xticks(rotation=90); 
plt.yticks(rotation=0); 
plt.title('Ind 0')
plt.show()

for i in [1,2,3,4]:
    plt.figure(figsize=(10,10))
    sns.clustermap(df_[df_['ind']==i][NamesCorr2].corr(),annot=True,
                cmap=plt.cm.jet,vmin=-.2,vmax=1,linewidths=.1); 
    plt.xticks(rotation=90); 
    plt.yticks(rotation=0); 
    plt.title('Ind '+str(i))
    plt.show()



for i in [0,1,2,3,4]:
    plt.figure(figsize=(6, 5))
    aaa=df_[df_['ind']==i]
    plt.scatter(aaa['H3K27ac'],aaa['H3K27me3'],s=3)
#    ax=sns.kdeplot(data=aaa,x='H3K27ac',y='H3K27me3',levels=10)
    R2=aaa[['H3K27ac','H3K27me3']].corr()
    plt.xlabel('H3K27ac');
    plt.ylabel('H3K27me3')
    plt.title("$T_"+str(i)+"$ R$^2$ = "+str(R2.iloc[0,1]))
    plt.show()
    
    
for i in [0,1,2,3,4]:
    plt.figure(figsize=(6, 5))
    aaa=df_[df_['ind']==i]
    plt.scatter(aaa['H3K27ac'],aaa['H3K36me2'],s=3)
#    ax=sns.kdeplot(data=aaa,x='H3K27ac',y='H3K27me3',levels=10)
    R2=aaa[['H3K27ac','H3K36me2']].corr()
    plt.xlabel('H3K27ac');
    plt.ylabel('H3K36me2')
    plt.title("$T_"+str(i)+"$ R$^2$ = "+str(R2.iloc[0,1]))
    plt.show()    

for i in [0,1,2,3,4]:
    plt.figure(figsize=(6, 5))
    aaa=df_[df_['ind']==i]
    plt.scatter(aaa['H3K27ac'],aaa['H3K9me3'],s=3)
#    ax=sns.kdeplot(data=aaa,x='H3K27ac',y='H3K27me3',levels=10)
    R2=aaa[['H3K27ac','H3K9me3']].corr()
    plt.xlabel('H3K27ac');
    plt.ylabel('H3K9me3')
    plt.title("$T_"+str(i)+"$ R$^2$ = "+str(R2.iloc[0,1]))
    plt.show()    

Names=['Cleaved H3',
 'H3K27me3',
 'H3K9me3',
 'H3.3',
 'H3K27ac',
 'H3K36me2',
 'H4K16ac',
 'H4',
 'H3K27M']

sns.set_style({'legend.frameon':True})

dd0=np.mean(d0[Names]).sort_values(ascending=False)
dd1=np.mean(d1[Names]).sort_values()
dd2=np.mean(d2[Names]).sort_values()
dd3=np.mean(d3[Names]).sort_values()
dd4=np.mean(d4[Names]).sort_values()
sz0=np.full(len(dd0),0,dtype=np.float64)
sz1=np.full(len(dd1),0,dtype=np.float64)
sz2=np.full(len(dd2),0,dtype=np.float64)
sz3=np.full(len(dd3),0,dtype=np.float64)
sz4=np.full(len(dd4),0,dtype=np.float64)


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
    q=np.std(d4[dd4.index[i]])*1/1.25
    sz4[i]=q

    
    
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.hlines(y=dd0.index, xmin=-5, xmax=5, color='gray', alpha=0.7, 
          linewidth=1, linestyles='dashdot')

ax.scatter(y=dd0.index, x=dd0, s=900*np.power(sz0,2), c='blue', alpha=0.7,
           label="0 hr",)
ax.scatter(y=dd1.index, x=dd1, s=900*np.power(sz1,2), c='green', alpha=0.7,
           label="8 hr",)
ax.scatter(y=dd2.index, x=dd2, s=900*np.power(sz2,2), c='yellow', alpha=0.7,
           label="16 hr",)
ax.scatter(y=dd3.index, x=dd3, s=900*np.power(sz3,2), c='orange', alpha=0.7,
           label="48 hr",)
ax.scatter(y=dd4.index, x=dd4, s=900*np.power(sz4,2), c='red', alpha=0.5,
           label="96 hr",)


ax.vlines(x=0, ymin=0, ymax=len(dd0)-1, color='black', alpha=0.7, linewidth=2, linestyles='dotted')
plt.legend(fontsize=30,
           facecolor='White', framealpha=1,frameon=True,
           loc='center right')

ax.set_title('Mean Value at Induction Time', fontdict={'size':30})
ax.set_xlim(-2.5, 2.5)

labels = dd0.index.to_list()
labels[8]="H3-K27M"
ax.set_yticklabels(labels)

plt.setp(ax.get_xticklabels(), fontsize=24)
plt.setp(ax.get_yticklabels(), fontsize=24)
plt.savefig("/Users/ronguy/Dropbox/Work/CyTOF/Plots/Induction_Mean.pdf")
plt.show()


dd0=np.mean(d0[Names_All]).sort_values(ascending=False)
dd1=np.mean(d1[Names_All]).sort_values()
dd2=np.mean(d2[Names_All]).sort_values()
dd3=np.mean(d3[Names_All]).sort_values()
dd4=np.mean(d4[Names_All]).sort_values()

for NN in Names_All:
    Var=NN
    TSNEVar=NN
    cc=ddf[TSNEVar]#[mask]
    cMeans=[dd0[Var],dd1[Var],dd2[Var],dd3[Var],dd4[Var]]
    cMeans=[i * 1 for i in cMeans]
    
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:,0],X_2d[:,1],s=2,
                c=cc, cmap=plt.cm.seismic)

#    plt.scatter(X,Y,s=150,alpha=1,
#                c=cMeans,edgecolors='black',cmap=plt.cm.jet)
#    for i in range(0,4):
#        plt.arrow(X[i], Y[i], dX[i], dY[i],length_includes_head=True,width=.1,color='black')

    plt.colorbar()
    plt.clim(-3.5,3.5)
    plt.clim(cc.quantile(0.01),cc.quantile(0.99))
    plt.savefig("/Users/ronguy/Dropbox/Work/CyTOF/Plots/Induction_"+NN+".png")
    plt.title(TSNEVar+" w/o K27M")












dfcorrcalc=df_.copy()
dfcorrcalc[df_mask]=float("nan")

d0=dfcorrcalc[dfcorrcalc['ind']==0].copy()
d4=dfcorrcalc[dfcorrcalc['ind']==4].copy()              


plt.figure(figsize=(10,10))
matrix=d4[NamesCorr1].corr()-d0[NamesCorr1].corr()
g=sns.clustermap(matrix, annot=True, annot_kws={"size":20},
        cmap=plt.cm.jet,vmin=matrix.min().min(),vmax=matrix.max().max(),linewidths=.1); 
plt.xticks(rotation=0); 
plt.yticks(rotation=0); 

plt.setp(g.ax_heatmap.get_xmajorticklabels(), fontsize = 20)
plt.setp(g.ax_heatmap.get_ymajorticklabels(), fontsize = 20)

g.ax_row_dendrogram.set_visible(False)
g.ax_col_dendrogram.set_visible(False)

g.ax_cbar.tick_params(labelsize=20)

plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)  # For y axis
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90) # For x axis
plt.title('Delta Correlations T4-T0')
plt.show()


plt.figure(figsize=(10,10))
matrix=d4[NamesCorr2].corr()-d0[NamesCorr2].corr()
g=sns.clustermap(matrix, annot=True,
        cmap=plt.cm.jet,vmin=matrix.min().min(),vmax=matrix.max().max(),linewidths=.1); 
plt.xticks(rotation=0); 
plt.yticks(rotation=0); 

plt.title('Delta Correlations T4-T0')
plt.show()



mask=(df_.ind==0) | (df_.ind==4)
BoxVar='c-Myc'
plt.figure(figsize=(6, 5))    
ax = sns.boxplot(x="ind", y=BoxVar, data=df_[mask],showfliers = False,palette=('blue','red'))
plt.title(BoxVar)




'''


NMS=[
'Cleaved H3',
 'H3K27me3',
 'H3K9me3',
 'H3.3',
 'H3K27ac',
 'H3K36me2',
 'H4K16ac',
 'H4',
 'H3K27M',
 'Ind'
]


random.shuffle(NMS)

MeansFrame=dd0.to_frame().transpose().append(dd1.to_frame().transpose()).append(dd2.to_frame().transpose())
MeansFrame=MeansFrame.append(dd3.to_frame().transpose()).append(dd4.to_frame().transpose())
MeansFrame=MeansFrame+10


MeansFrame=MeansFrame.assign(Ind=[0,1,2,3,4])

plt.figure(figsize=(30,30))
ax=pd.plotting.radviz(MeansFrame[NMS],class_column='Ind',
                     color=['blue','green','yellow','orange','red'],
                     s=1500,alpha=1)

ax.legend( loc='center left', bbox_to_anchor=(-.1, 1),
               fontsize=50, fancybox=True, ncol=5 ,markerscale=1.5)
ax.set_title( 'Ind', loc='right', fontsize=50,color='black' )
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

#for p in ax.patches[1:]:
#    p.set_visible(True)
#    p.set_color('black')

ax.patches[0].set_color('green')
ax.patches[0].set_alpha(0.1)

for t in ax.texts:
    t.set_color('black')
    t.set_fontsize(50)
    
plt.show()
'''