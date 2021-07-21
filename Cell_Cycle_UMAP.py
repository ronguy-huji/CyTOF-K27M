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
import joypy
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
import matplotlib
from scipy.spatial import ConvexHull

from pandas.io.formats.printing import pprint_thing
import matplotlib.lines as mlines
import matplotlib.patches as patches

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

#df=pd.read_csv("control.csv")
#dfmut=pd.read_csv("mutant.csv")
ExpName="C17"
dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets/17321/"
df=pd.read_csv(dir+ExpName+".csv")


import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)



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
    # ['H3',
      'IdU',
    #  'MBP',
    #  'H3K36me3',
    #  'GFAP',
    #  'EZH2',
    #  'H3K4me3',
    #  'H3K79me2',
    #  'pHistone H2A.X [Ser139]',
    #  'H3K36me2',
    #  'Sox2',
    #  'SIRT1',
    #  'H4K16ac',
    #  'H2AK119ub',
    #  'H3K4me1',
    #  'H3.3',
    #  'H3K64ac',
    #  'BMI1',
    #  'Cmyc',
    #  'H4',
    #  'H3K27ac',
    #  'H4K20me3',
    #  'DLL3',
    #  'cleaved H3',
    #  'H3K9ac',
    #  'H1.0',
    #  'CD24',
    #  'H3K27me3',
    #  'H3K27M',
    #  'H3K9me3',
    #  'CD44',
    #  'Ki-67',
    #  'CXCR4',
    #  'pHistone H3 [S28]',
    # 'H1'
]

plt.figure(figsize=(6, 5))




GateColumns = ['H3',
 'H3K4me3',
 'H3K79me2',
 'H3K36me2',
 'H4K16ac',
 'H2Aub',
 'H3K4me1',
 'H3.3',
 'H3K64ac',
 'H4',
 'H4K20me3',
 'H3K9ac',
 'H3K27ac',
 'H3K27me3',
 'H3K27M',
 'H3K9me3'
 ]


BoxList=[
 'H3',
 'IdU',
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

df=df[(df[['H3','H3.3','H4']]>5).all(axis=1)]
df=df[(df[GateColumns]>0).all(axis=1)]


ldf=len(df)
print(ExpName)
for col in df.columns:
    lG=  np.count_nonzero(df[col]<2)  
    print("%-30s %d %5.3f" % (col,lG,lG/ldf*100.)) 
print(" ")


dfmask=df<2
dfmask['H3K27M']=df['H3K27M']<150
dfmask['IdU']=df['IdU']<150







ldf=len(df)
print(ExpName)
for col in df.columns:
    lG=  (dfmask[col].sum())  
    print("%-30s %d %5.3f" % (col,lG,lG/ldf*100.)) 
print(" ")

dfpre=df

scFac=5
df=np.arcsinh(df/scFac)

mask=dfmask['IdU']
### Diff Plot Pre-Normalization

### Diff Plot


DiffMarks=[
 'H3K36me3',
 'H3K4me3',
 'H3K79me2',
 'yH2A.X',
 'H3K36me2',
 'H4K16ac',
 'H2Aub',
 'H3K4me1',
 'H3K64ac',
 'H3K27ac',
 'H4K20me3',
 'cleaved H3',
 'H3K9ac',
 'H1.0',
 'H3K27me3',
# 'H3K27M',
 'H3K9me3',
 'pH3[S28]',
 'H1.3/4/5', 'H3','H4','H3.3'
]

d0=df[mask].copy()
d1=df[~mask].copy()
dd0=np.mean(d0[NamesAll]).sort_values(ascending=False)
dd1=np.mean(d1[NamesAll]).sort_values()

#dd0=dd0.drop(labels=['H3','H3.3','H4'])
#dd1=dd1.drop(labels=['H3','H3.3','H4'])
diffs=(dd1-dd0).sort_values(ascending=False)    
diffs=diffs[DiffMarks].sort_values(ascending=False) 
colors = ['dodgerblue' if x < 0 else 'darkmagenta' for x in diffs]

# Draw plot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=1, linewidth=5)

# Decorations
plt.gca().set(ylabel='', xlabel='')
#plt.yticks(df.index, df.cars, fontsize=12)
plt.xticks(fontsize=30 ) 
plt.yticks(fontsize=30 ) 
plt.title(ExpName+' IdU Clusters - Pre Norm Pre Standard', fontdict={'size':20})
plt.grid(linestyle='--', alpha=0.5)
plt.show()

####


df_Bck=df.copy()
df2=df.copy()
aaaa=df
m=np.mean(aaaa)
s=np.std(aaaa)


### Diff Plot Pre-Normalization

### Diff Plot


DiffMarks=[
 'H3K36me3',
 'H3K4me3',
 'H3K79me2',
 'yH2A.X',
 'H3K36me2',
 'H4K16ac',
 'H2Aub',
 'H3K4me1',
 'H3K64ac',
 'H3K27ac',
 'H4K20me3',
 'cleaved H3',
 'H3K9ac',
 'H1.0',
 'H3K27me3',
# 'H3K27M',
 'H3K9me3',
 'pH3[S28]',
 'H1.3/4/5', 'H3','H4','H3.3'
]
df2=(df2-m)/s
d0=df2[mask].copy()
d1=df2[~mask].copy()
dd0=np.mean(d0[NamesAll]).sort_values(ascending=False)
dd1=np.mean(d1[NamesAll]).sort_values()

#dd0=dd0.drop(labels=['H3','H3.3','H4'])
#dd1=dd1.drop(labels=['H3','H3.3','H4'])
diffs=(dd1-dd0).sort_values(ascending=False)    
diffs=diffs[DiffMarks].sort_values(ascending=False) 
colors = ['dodgerblue' if x < 0 else 'darkmagenta' for x in diffs]

# Draw plot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=1, linewidth=5)

# Decorations
plt.gca().set(ylabel='', xlabel='')
#plt.yticks(df.index, df.cars, fontsize=12)
plt.xticks(fontsize=30 ) 
plt.yticks(fontsize=30 ) 
plt.title(ExpName+' IdU Clusters - Pre Norm After Standard', fontdict={'size':20})
plt.grid(linestyle='--', alpha=0.5)
plt.show()

####






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
    out = minimize(residual, params, args=(aaaa, aaaa),method='cg')
    alpha=out.params['alpha'].value
    beta=out.params['beta'].value
    gam=out.params['gamma'].value
    
    ddf=df
    ddf=(ddf-m)/s

    avMarkers=ddf['H3.3']*alpha+ddf['H4']*beta+ddf['H3']*gam
    
    
    ddf=ddf.subtract(avMarkers,axis=0)
#    ddf[NMS]=C18_Bck[NMS]
    df=ddf

#C18=(C18-m)/s
df_Clust=df.copy()

for NN in Names_UMAP:    
    df_Clust.loc[dfmask[NN],NN]=0
    df_Clust.loc[~dfmask[NN],NN]=1
    
### tSNE C18


#X_2d = tsne.fit_transform(ar)#[mask,:])

X_2d=draw_umap(df_Clust[Names_UMAP],cc=df['IdU'],min_dist=0.1,n_neighbors=20)

df_bck=df
#C18[dfmask]=-100

for NN in Names_UMAP:#tSNE:
    Var=NN
    TSNEVar=NN
    cc=df[TSNEVar]#[mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:,0],X_2d[:,1],s=2,
                c=cc, cmap=plt.cm.jet)

    plt.colorbar()
    plt.clim(-3.5,3.5)
    cmap = matplotlib.cm.get_cmap('jet')
    mask=dfmask[TSNEVar]==True
    rgba = cmap(-10)
    plt.scatter(X_2d[mask][:,0],X_2d[mask][:,1],s=2,
                color=rgba) 
    plt.title(TSNEVar+" "+ExpName+" Cell Cycle")



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

df=df.assign(umap0=X[:,0])
df=df.assign(umap1=X[:,1])
lab=dbscan_plot(X_2d,eps=0.2,min_samples=50)

#C18=C18.assign(clust=lab)
df=df.assign(clust=lab)
df_Clust=df_Clust.assign(clust=lab)

df=df[df['clust']!=-1]
df_Clust=df_Clust[df_Clust['clust']!=-1]

print(df_Clust[Names_UMAP+['clust']].groupby('clust').mean())
sns.heatmap(df_Clust[Names_UMAP+['clust']].groupby('clust').mean(),cmap=plt.cm.jet)


for NN in BoxList:
     BoxVar=NN
     plt.figure(figsize=(3, 5))    
     ax = sns.boxplot(x="clust", y=NN, data=df,showfliers=False,palette=['dodgerblue','darkmagenta'])
     ax.set_xticklabels(['IdU \nLow','IdU \nHigh'])

     plt.show()





    
NamesPair=[
    #  'IdU',
    #  'MBP',
      'H3K36me3',
    #  'GFAP',
    #  'EZH2',
      'H3K4me3',
      'H3K79me2',
    #  'pHistone H2A.X [Ser139]',
      'H3K36me2',
    #  'Sox2',
    #  'SIRT1',
      'H4K16ac',
      'H2Aub',
      'H3K4me1',
    #  'H3.3',
      'H3K64ac',
    #  'BMI1',
    #  'Cmyc',
    #  'H4',
      'H3K27ac',
      'H4K20me3',
    #  'DLL3',
      'cleaved H3',
      'H3K9ac',
      'H1.0',
    #  'CD24',
      'H3K27me3',
      'H3K27M',
      'H3K9me3',
    #  'CD44',
    #  'Ki-67',
    #  'CXCR4',
    #  'pHistone H3 [S28]',
    #  'H1',
    'clust'
]
    
# sns.pairplot(data=df[NamesPair],hue='clust',palette='tab10',
#              corner=True,plot_kws={"s": 3},
#              diag_kind="kde",diag_kws=dict(fill=True, common_norm=False),)



features=[
 'H3K9ac',
 'H3K27ac',
 'H3K64ac',
 'cleaved H3',
 'H3K27M',
 'clust'
]
    



mask=df.clust.isin([0,1])


    
#Rewriet RadViz function
    
def encircle(x,y,ax = None ,**kw):
    if not ax: ax = plt.gca()
    p = np.c_[x,y]

    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices,:],**kw)
    ax.add_patch(poly)
    
   
def radviz2(
    frame,
    class_column,
    ax = None,
    color=None,
    colormap=None,
    **kwds,
):
    import matplotlib.pyplot as plt

    def normalize(series):
        a = min(series)
        b = max(series)
        return (series - a) / (b - a)

    n = len(frame)
    classes = frame[class_column].drop_duplicates()
    class_col = frame[class_column]
    df = frame.drop(class_column, axis=1).apply(normalize)

    if ax is None:
        ax = plt.gca()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    to_plot: dict[Hashable, list[list]] = {}
#    colors = get_standard_colors(
#        num_colors=len(classes), colormap=colormap, color_type="random", color=color
#    )
    colors=['dodgerblue','darkmagenta']
    for kls in classes:
        to_plot[kls] = [[], []]

    m = len(frame.columns) - 1
    s = np.array(
        [(np.cos(t), np.sin(t)) for t in [2 * np.pi * (i / m) for i in range(m)]]
    )

    for i in range(n):
        row = df.iloc[i].values
        row_ = np.repeat(np.expand_dims(row, axis=1), 2, axis=1)
        y = (s * row_).sum(axis=0) / row.sum()
        kls = class_col.iat[i]
        to_plot[kls][0].append(y[0])
        to_plot[kls][1].append(y[1])
        # if (-0.2<y[0]<0.2) & (-0.2<y[1]<0.2):
        #     to_plot[kls][0].append(y[0])
        #     to_plot[kls][1].append(y[1])

    for i, kls in enumerate(classes):
        ax.scatter(
            to_plot[kls][0],
            to_plot[kls][1],
            color=colors[i],
            label=pprint_thing(kls),
            **kwds,
        )
        # encircle(to_plot[kls][0],
        #     to_plot[kls][1], ax=ax,
        #     ec = 'k',fc = colors[i],
        #     alpha = 0.2)
        # ax.scatter(
        #     np.asarray(to_plot[kls][0]).mean(),
        #     np.asarray(to_plot[kls][1]).mean(),
        #     color=colors[i], alpha=1,
        #     label=pprint_thing(kls),edgecolor='k', linewidth=3,s=200)
        # sns.kdeplot(x=np.asarray(to_plot[kls][0]),
        #             y=np.asarray(to_plot[kls][1]),color=colors[i],ax=ax,bw_adjust=1,
        #             levels=5)
        # ax.scatter(
        #      np.asarray(to_plot[kls][0]).mean(),
        #      np.asarray(to_plot[kls][1]).mean(),
        #      color=colors[i], alpha=1,
        #      label=pprint_thing(kls),edgecolor='k', linewidth=3,s=200)

            
        

        
    ax.legend()

    ax.add_patch(patches.Circle((0.0, 0.0), radius=1.0, facecolor="none",edgecolor='black'))
    s=s*1.
    for xy, name in zip(s, df.columns):
        rad=0.05
        ax.add_patch(patches.Circle(xy, radius=rad, facecolor="black"))

        if xy[0] < 0.0 and xy[1] < 0.0:
            ax.text(
                xy[0] - rad, xy[1] - rad, name, ha="right", va="top", size="small"
            )
        elif xy[0] < 0.0 and xy[1] >= 0.0:
            ax.text(
                xy[0] - rad,
                xy[1] + rad,
                name,
                ha="right",
                va="bottom",
                size="small",
            )
        elif xy[0] >= 0.0 and xy[1] < 0.0:
            ax.text(
                xy[0] + rad, xy[1] - rad, name, ha="left", va="top", size="small"
            )
        elif xy[0] >= 0.0 and xy[1] >= 0.0:
            ax.text(
                xy[0] + rad, xy[1] + rad, name, ha="left", va="bottom", size="small"
            )

    ax.axis("equal")
    return ax, to_plot    
### End Radviz
##### 

from histEq import *

features=[
    'H3K27ac', 
    'H3K9ac',
    'H3K64ac', 
    'H2Aub', 
#    'H4K16ac',
    'H3K9me3'
    ]    
    
ddf=1*(df.sample(frac=1).sort_values('clust',ascending=True))

dfmean=df.groupby('clust').mean()
dfmean=dfmean.assign(clust=[0,1])



dff=(df-df.min())/(df.max()-df.min())
dff=dff.assign(clust=df.clust)

for NN in features:
    dff[NN]=Equalize(dff[NN],256)/256

plt.figure(figsize=(10,10))
ax,tp=radviz2(dff[features+['clust']],class_column='clust',
                     color=['blue','red'],
                     s=20,alpha=0.3)


x0=np.asarray(tp[0][0]); 
y0=np.asarray(tp[0][1]); 
x0 = x0[~np.isnan(x0)]
y0 = y0[~np.isnan(y0)]

x1=np.asarray(tp[1][0]); 
y1=np.asarray(tp[1][1]); 

xy = np.vstack([x0,y0])
pts0 = xy.mean(axis=1)#xy.T[np.argmax(density)]

xy = np.vstack([x1,y1])
pts1 = xy.mean(axis=1)#xy.T[np.argmax(density)]





ax.legend( loc='center left', bbox_to_anchor=(1.25, 1),
               fontsize=30, fancybox=True, ncol=1 ,markerscale=2)
ax.set_title( ExpName, loc='right', fontsize=30,color='black' )
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
SP=['left','right','top','bottom']
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)



for t in ax.texts:
    t.set_color('black')
    t.set_fontsize(30)
ax.scatter(pts1[0],pts1[1],color='darkmagenta',s=300,ec='k',linewidth=4,alpha=1,zorder=100);
ax.scatter(pts0[0],pts0[1],color='dodgerblue',s=300,ec='k',linewidth=4,alpha=1,zorder=105);
#ax.scatter(0,0,color='black',s=200,ec='k',linewidth=3,alpha=1,zorder=115);
plt.show()

x0=np.asarray(tp[0][0]).astype(np.float16); 
y0=np.asarray(tp[0][1]).astype(np.float16); 
x1=np.asarray(tp[1][0]).astype(np.float16); 
y1=np.asarray(tp[1][1]).astype(np.float16); 
r0=np.sqrt(x0*x0+y0*y0);
r1=np.sqrt(x1*x1+y1*y1);
r = np.concatenate((r0, r1))
r_new=Equalize(r,256)/256;
x0_new=r_new[0:len(x0)]*x0/r0
y0_new=r_new[0:len(x0)]*y0/r0
x1_new=r_new[0:len(x1)]*x1/r1
y1_new=r_new[0:len(x1)]*y1/r1

####

features=[
    'H3K27ac', 
    'H3K9ac',
    'H3K64ac', 
    'H2Aub', 
    'H4K16ac',
    'H3K36me3',
    'H3K9me3'
    ]    
    
ddf=1*(df.sample(frac=1).sort_values('clust',ascending=True))

dfmean=df.groupby('clust').mean()
dfmean=dfmean.assign(clust=[0,1])



dff=(df-df.min())/(df.max()-df.min())
dff=dff.assign(clust=df.clust)

for NN in features:
    dff[NN]=Equalize(dff[NN],256)/256

plt.figure(figsize=(10,10))
ax,tp=radviz2(dff[features+['clust']],class_column='clust',
                     color=['blue','red'],
                     s=20,alpha=0.3)


x0=np.asarray(tp[0][0]); 
y0=np.asarray(tp[0][1]); 
x1=np.asarray(tp[1][0]); 
y1=np.asarray(tp[1][1]); 

xy = np.vstack([x0,y0])
kde = stats.gaussian_kde(xy)
density = kde(xy)
pts0 = xy.mean(axis=1)#xy.T[np.argmax(density)]

xy = np.vstack([x1,y1])
kde = stats.gaussian_kde(xy)
density = kde(xy)
pts1 = xy.mean(axis=1)#xy.T[np.argmax(density)]





ax.legend( loc='center left', bbox_to_anchor=(1.25, 1),
               fontsize=30, fancybox=True, ncol=1 ,markerscale=2)
ax.set_title( ExpName, loc='right', fontsize=30,color='black' )
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
SP=['left','right','top','bottom']
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)



for t in ax.texts:
    t.set_color('black')
    t.set_fontsize(30)
ax.scatter(pts1[0],pts1[1],color='darkmagenta',s=300,ec='k',linewidth=4,alpha=1,zorder=100);
ax.scatter(pts0[0],pts0[1],color='dodgerblue',s=300,ec='k',linewidth=4,alpha=1,zorder=105);
#ax.scatter(0,0,color='black',s=200,ec='k',linewidth=3,alpha=1,zorder=115);
plt.show()

x0=np.asarray(tp[0][0]).astype(np.float16); 
y0=np.asarray(tp[0][1]).astype(np.float16); 
x1=np.asarray(tp[1][0]).astype(np.float16); 
y1=np.asarray(tp[1][1]).astype(np.float16); 
r0=np.sqrt(x0*x0+y0*y0);
r1=np.sqrt(x1*x1+y1*y1);
r = np.concatenate((r0, r1))
r_new=Equalize(r,256)/256;
x0_new=r_new[0:len(x0)]*x0/r0
y0_new=r_new[0:len(x0)]*y0/r0
x1_new=r_new[0:len(x1)]*x1/r1
y1_new=r_new[0:len(x1)]*y1/r1




### Diff Plot


DiffMarks=[
 'H3K36me3',
 'H3K4me3',
 'H3K79me2',
 'yH2A.X',
 'H3K36me2',
 'H4K16ac',
 'H2Aub',
 'H3K4me1',
 'H3K64ac',
 'H3K27ac',
 'H4K20me3',
 'cleaved H3',
 'H3K9ac',
 'H1.0',
 'H3K27me3',
# 'H3K27M',
 'H3K9me3',
 'pH3[S28]',
 'H1.3/4/5'
]

d0=df[df.clust==0]
d1=df[df.clust==1]
dd0=np.mean(d0[NamesAll]).sort_values(ascending=False)
dd1=np.mean(d1[NamesAll]).sort_values()

dd0=dd0.drop(labels=['H3','H3.3','H4'])
dd1=dd1.drop(labels=['H3','H3.3','H4'])
diffs=(dd1-dd0).sort_values(ascending=False)    
diffs=diffs[DiffMarks].sort_values(ascending=False) 
colors = ['dodgerblue' if x < 0 else 'darkmagenta' for x in diffs]

# Draw plot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
plt.hlines(y=diffs.index, xmin=0, xmax=diffs, color=colors, alpha=1, linewidth=5)

# Decorations
plt.gca().set(ylabel='', xlabel='')
#plt.yticks(df.index, df.cars, fontsize=12)
plt.xticks(fontsize=30 ) 
plt.yticks(fontsize=30 ) 
plt.title(ExpName+' IdU Clusters', fontdict={'size':20})
plt.grid(linestyle='--', alpha=0.5)
plt.show()




def estimate_maxima(data):

      kde = gaussian_kde(data)

      no_samples = 10

      samples = np.linspace(0, 10, no_samples)

      probs = kde.evaluate(samples)

      maxima_index = probs.argmax()

      maxima = samples[maxima_index]

      return maxima
  
    

# plt.show()
#######


features=['H3',
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
 'H3K64ac',
 'BMI1',
 'Cmyc',
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
    
ddf=1*(df.sample(frac=1).sort_values('clust',ascending=True))

dff=(df-df.min())/(df.max()-df.min())
dff=dff.assign(clust=df.clust)

for NN in features:
    dff[NN]=Equalize(dff[NN],256)/256



gg=dff[features].copy().assign(clust=df.clust)
#gg=(gg-gg.min())/(gg.max()-gg.min())
# for NN in features:
#     gg[NN]=stats.boxcox(gg[NN]+0.001)[0]
#gg=(gg-gg.mean())/(gg.std())
r=gg.groupby('clust').median()

r_min=gg.groupby('clust').quantile(0.25)
r_max=gg.groupby('clust').quantile(0.75)


NumFeat=len(features)

r=r[features]
r_min=r_min[features]
r_max=r_max[features]
r=r.assign(end=r.iloc[:,0])
r_min=r_min.assign(end=r_min.iloc[:,0])
r_max=r_max.assign(end=r_max.iloc[:,0])
theta = [2 * np.pi * (i / NumFeat) for i in range(NumFeat+1)]

fig = plt.figure(1)
ax1 = fig.add_subplot(111, projection='polar')
ax1.xaxis.labelpad = 100
plt.polar(theta,r.iloc[0,:],color='blue')
plt.polar(theta,r_min.iloc[0,:],color='blue',linestyle='dashed')
plt.polar(theta,r_max.iloc[0,:],color='blue',linestyle='dashed')
plt.fill_between(theta, r_min.iloc[0,:], r_max.iloc[0,:], alpha=0.2,color='blue')
plt.polar(theta,r.iloc[1,:],color='red')
plt.polar(theta,r_min.iloc[1,:],color='red',linestyle='dashed')
plt.polar(theta,r_max.iloc[1,:],color='red',linestyle='dashed')
plt.fill_between(theta, r_min.iloc[1,:], r_max.iloc[1,:], alpha=0.2,color='red')
#plt.yticks(ticks=None)
plt.xticks(ticks=None)
#ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.set_xticks(theta[0:NumFeat])
ax1.set_xticklabels(features)
ax1.set_rmax(1.0)

for theta, label in zip(ax1.get_xticks(), ax1.get_xticklabels()):
        theta = theta * ax1.get_theta_direction() + ax1.get_theta_offset()
        theta = np.pi/2 - theta
        y, x = np.cos(theta), np.sin(theta)
        if x >= 0.1:
            label.set_horizontalalignment('left')
        if x <= -0.1:
            label.set_horizontalalignment('right')
        if y >= 0.5:
            label.set_verticalalignment('bottom')
        if y <= -0.5:
            label.set_verticalalignment('top')

plt.show()
import math
def sigmoid(x,scale=1,center=0):
  return 1 / (1 + np.exp(-(x-center)/scale))


from skimage import exposure
dff=(df-df.min())/(df.max()-df.min())
dff=dff.assign(clust=df.clust)
dff.sort_values('clust',inplace=True)

bins=256

for NN in features:
    img_or=np.asarray(dff[NN])
    img=np.asarray(np.round(img_or*(bins-1))).astype(np.uint16)
    # dff[NN] = exposure.equalize_adapthist(img,
    #                                       kernel_size=dff.groupby('clust').count().min().min(),
    #                                       clip_limit=0.03)
    dff[NN] = exposure.equalize_hist(img)
     

gg=dff[features].copy().assign(clust=df.clust)
#gg=(gg-gg.min())/(gg.max()-gg.min())


# for NN in features:
#     gg[NN]=stats.boxcox(gg[NN]+0.001)[0]
#gg=(gg-gg.mean())/(gg.std())
r=gg.groupby('clust').median()

r_min=gg.groupby('clust').quantile(0.25)
r_max=gg.groupby('clust').quantile(0.75)


NumFeat=len(features)

r=r[features]
r_min=r_min[features]
r_max=r_max[features]
r=r.assign(end=r.iloc[:,0])
r_min=r_min.assign(end=r_min.iloc[:,0])
r_max=r_max.assign(end=r_max.iloc[:,0])
theta = [2 * np.pi * (i / NumFeat) for i in range(NumFeat+1)]

fig = plt.figure(1)
ax1 = fig.add_subplot(111, projection='polar')
ax1.xaxis.labelpad = 100
plt.polar(theta,r.iloc[0,:],color='blue')
plt.polar(theta,r_min.iloc[0,:],color='blue',linestyle='dashed')
plt.polar(theta,r_max.iloc[0,:],color='blue',linestyle='dashed')
plt.fill_between(theta, r_min.iloc[0,:], r_max.iloc[0,:], alpha=0.2,color='blue')
plt.polar(theta,r.iloc[1,:],color='red')
plt.polar(theta,r_min.iloc[1,:],color='red',linestyle='dashed')
plt.polar(theta,r_max.iloc[1,:],color='red',linestyle='dashed')
plt.fill_between(theta, r_min.iloc[1,:], r_max.iloc[1,:], alpha=0.2,color='red')
#plt.yticks(ticks=None)
plt.xticks(ticks=None)
#ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.set_xticks(theta[0:NumFeat])
ax1.set_xticklabels(features)
ax1.set_rmax(1.0)

for theta, label in zip(ax1.get_xticks(), ax1.get_xticklabels()):
        theta = theta * ax1.get_theta_direction() + ax1.get_theta_offset()
        theta = np.pi/2 - theta
        y, x = np.cos(theta), np.sin(theta)
        if x >= 0.1:
            label.set_horizontalalignment('left')
        if x <= -0.1:
            label.set_horizontalalignment('right')
        if y >= 0.5:
            label.set_verticalalignment('bottom')
        if y <= -0.5:
            label.set_verticalalignment('top')

plt.show()

