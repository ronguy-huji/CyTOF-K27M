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
    u1  = di.StringItem("Clusters to keep", default='')
    u2  = di.StringItem("Clusters to dump", default='')
#    u3  = di.StringItem("Cluster 3", default='')
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
dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets/"
F1="C05"
F2="C07"
df=pd.read_csv(dir+F1+".csv")
dfmut=pd.read_csv(dir+F2+".csv")



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
    params.add('alpha', value=0.5, min=0)
    params.add('gamma', value=0.1, min=0)
    aaaa=(aaaa-m)/s
    out = minimize(residual, params, args=(aaaa, aaaa),method='cg')
    alpha=out.params['alpha'].value
    beta=out.params['beta'].value
    gam=out.params['gamma'].value
    
    ddf=dfmut
    ddf=(ddf-m)/s
    avMarkers=ddf['H3.3']*alpha+ddf['H4']*beta+ddf['H3']*gam
    ddf=ddf.subtract(avMarkers,axis=0)
    dfmut=ddf

    ddf=df
    ddf=(ddf-m)/s
    avMarkers=ddf['H3.3']*alpha+ddf['H4']*beta+ddf['H3']*gam
    ddf=ddf.subtract(avMarkers,axis=0)
    df=ddf






####

ddf=dfmut.sample(frac=1)

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

### DBSCAN Section


####
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




df_ = pd.DataFrame(columns=list(dfmut.columns))


for i in Clus1:
    mask=(labels==i)
    ddd=dfmut[mask]#.assign(clus=1)
    df_=df_.append(ddd)
    
df_=df_.assign(Type='Mutant')
aaaa=df_.append(df.assign(Type='Wild'))
   


''' 
  
for NN in Names:
    BoxVar=NN
    plt.figure(figsize=(6, 5))    
    ax = sns.boxplot(x="Type", y=NN, data=aaaa.sort_values("Type",ascending=False),showfliers=False)
    plt.title(NN)
'''

CutVar='pHistone H3 [S28]'

Top=df[df[CutVar]> df[CutVar].quantile(0.9)]
#Mid=dfmut[((dfmut[CutVar] > dfmut[CutVar].quantile(0.25)) & 
#           (dfmut[CutVar] < dfmut[CutVar].quantile(0.75)))]
Bottom=df[df[CutVar] < df[CutVar].quantile(0.9)]
pH=Bottom.assign(Top=False)
pH=pH.append(Top.assign(Top=True))
'''
for NN in Names:
    BoxVar=NN
    plt.figure(figsize=(6, 5))    
    ax = sns.boxplot(x="Top", y=NN, data=pH.sort_values("Top",ascending=False),showfliers=False)
    plt.title(NN+' Cut on '+CutVar)
'''    




#df_.to_csv(param.case+'.csv')

'''
Var1='H3K27ac'
Var2='H3K9ac'
plt.figure(figsize=(6, 5))    
plt.scatter(df[Var1],df[Var2],s=2); 
plt.ylim(-4,4);
plt.xlim(-4,4);
plt.xlabel(Var1);
plt.ylabel(Var2);

Var1='H3K27ac'
Var2='H4K16Ac'
plt.figure(figsize=(6, 5))    
plt.scatter(df[Var1],df[Var2],s=2); 
plt.ylim(-4,4);
plt.xlim(-4,4);
plt.xlabel(Var1);
plt.ylabel(Var2);
'''

title='C15'

Vars=['H3K27ac','H3K9ac','H3K64ac','H4K16Ac']
plt.figure(figsize=(6, 5))    
sns.pairplot(df[Vars],corner=True,plot_kws={"s": 3})
plt.title(title)

plt.figure(figsize=(6, 5))    
sns.heatmap(dfmut[Vars].corr(),annot=True,cmap=plt.cm.jet,vmin=0,vmax=1,linewidths=.1); 
plt.xticks(rotation=90); 
plt.yticks(rotation=0); 
plt.show()
'''

Vars=['H3K4me3','H3K4me1','H4K16Ac','H3K9ac','H3K64ac','H3K27ac']
plt.figure(figsize=(6, 5)) 
sns.pairplot(df[Vars],corner=True,plot_kws={"s": 3})
plt.title(title)

### SAME for C13

Vars=['H3K9me3','H3K27me3', 'H3K36me2', 'H2AK119ub', 'H3K9ac', 'H3K27ac']
plt.figure(figsize=(6, 5))    
sns.pairplot(df[Vars],corner=True,plot_kws={"s": 3})
plt.title(title)

'''
## SAME for C13
size=min(len(dfmut),len(df))
EqSize=aaaa[aaaa['Type']=="Mutant"].iloc[0:size]
EqSize=EqSize.append(aaaa[aaaa['Type']=="Wild"].iloc[0:size])


title='C15/C16'
Vars=['H3K27ac','H4K16Ac','H3K4me1','H3K4me3','H2AK119ub','H3K27me3','Type']
plt.figure(figsize=(6, 5))    
sns.pairplot(EqSize[Vars].sort_values("Type",ascending=False),
             corner=True,hue="Type",plot_kws={"s": 3},
        )
plt.title(title)



'''
Vars=['H3K27ac','H4K16Ac','H3K4me1','H3K4me3','H2AK119ub','H3K27me3']
plt.figure(figsize=(6, 5))    
sns.pairplot(dfmut[Vars],corner=True,plot_kws={"s": 3})
plt.title("C16")
'''

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

sns.set_style({'legend.frameon':True})
df=np.mean(aaaa[aaaa['Type']=="Mutant"][Names]).sort_values()
s1=np.full((len(df),4),0,dtype=np.float64)
sz1=np.full(len(df),0,dtype=np.float64)
df2=np.mean(aaaa[aaaa['Type']=="Wild"][Names]).sort_values()
s2=np.full((len(df2),4),0,dtype=np.float64)
sz2=np.full(len(df2),0,dtype=np.float64)

for i in range(0,len(df)):
    print(str(i)+" "+df.index[i])
    q=np.std(aaaa[aaaa['Type']=="Mutant"][df.index[i]])*1/1.25
    print(q)
    s1[i]=[float(q),0,0,q]
    sz1[i]=q
    print(str(i)+" "+df2.index[i])
    q=np.std(aaaa[aaaa['Type']=="Wild"][df2.index[i]])*1./1.25
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

ax.set_title('Mean of Distribution - '+F1+'/'+F2, fontdict={'size':22})
ax.set_xlim(-1.5, 1.5)

plt.show()

'''

sns.set_style({'legend.frameon':True})
df=aaaa[aaaa['Type']=="Mutant"][Names].median().sort_values()
s1=np.full(len(df),0,dtype=np.float64)*255./1.5
df2=aaaa[aaaa['Type']=="Wild"][Names].median().sort_values()
s2=np.full(len(df2),0,dtype=np.float64)

for i in range(0,len(df)):
    print(str(i)+" "+df.index[i])
    q=np.std(aaaa[aaaa['Type']=="Mutant"][df.index[i]])
    print(q)
    s1[i]=float(q)
    print(str(i)+" "+df2.index[i])
    q=np.std(aaaa[aaaa['Type']=="Wild"][df2.index[i]])
    print(q)
    s2[i]=float(q)
    
    
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.hlines(y=df.index, xmin=-5, xmax=5, color='gray', alpha=0.7, 
          linewidth=1, linestyles='dashdot')
ax.scatter(y=df.index, x=df, s=200, color='firebrick', alpha=0.7,
           label="Mutant",)
#ax.hlines(y=df.index, xmin=df-s1, xmax=df+s1, color='firebrick', alpha=0.7, 
#          linewidth=3, linestyles='dashdot')
ax.scatter(y=df2.index, x=df2, s=200, color='blue', alpha=0.7,
           label="Wild")
#ax.hlines(y=df2.index, xmin=df2-s2, xmax=df2+s2, color='blue', alpha=0.7, 
#          linewidth=3, linestyles='dashdot')
ax.vlines(x=0, ymin=0, ymax=len(df)-1, color='black', alpha=0.7, linewidth=2, linestyles='dotted')
plt.legend(fontsize=20,
           facecolor='White', framealpha=1,frameon=True)
ax.set_title('Median of Distribution', fontdict={'size':22})
ax.set_xlim(-2.5, 2.5)

plt.show()
'''
'''
NMS=[
     'H3K4me1', 'H3K4me3', 'H4K16Ac', 'H3K9ac', 'H3K64ac', 'H3K27ac'
     ]
NMS2=['H3K9me3', 'H3K27me3', 'H3K36me2', 'H2AK119ub', 'H3K9ac']

sns.heatmap(dfmut[Names].corr()-df[Names].corr(),annot=True,cmap=plt.cm.jet,vmin=0,vmax=1,linewidths=.1);
plt.xticks(rotation=90); 
plt.yticks(rotation=0)
plt.title("C15"); 
plt.show()
'''

#H3K4me1, H3K4me3, H4K16ac, H3K9ac H3K64ac, H3K27ac
from chord import Chord

ChordVars=[
 'H3K27ac',
 'H3K4me1',
 'H3K4me3',
 'H3K64ac',
 'H3K9ac',
 'H4K16Ac'
 ]
'''
Chord.user = "guy.ron@gmail.com"
Chord.key = "CP-2bd965c5-15f8-4293-8e61-94c9f7f06821"
import holoviews as hv
qq=aaaa[aaaa['Type']=="Wild"]
matrix = np.power(qq[ChordVars].corr(),1)
matrix[matrix < 0] = 0
matrix[matrix==1]=0
matrix=(10*matrix).astype('int')
matrix.to_csv("m.csv")

df=pd.read_csv("m.csv")
df = df.set_index('Unnamed: 0')
df.index.name = None
links = hv.Dataset((list(df.columns), list(df.index), df),
                  ['source', 'target'], 'value').dframe()

nodes = hv.Dataset(nd, 'index')
nodes.data.head()


chord = hv.Chord((links,nodes)).select(value=(1, None))
chord.opts(
    opts.Chord(cmap='Category10', edge_cmap='Category10', edge_color=dim('source').str(), 
               labels='name', node_color=dim('index').str()))


#chord.opts(
#          opts.Chord(cmap='Category10', edge_cmap='Category10', edge_color='source', 
#                           labels='name', node_color=dim('group').str()))
show(hv.render(chord))

#chord.opts(
#    node_color='index', edge_color='source', label_index='index', 
#    cmap='Category10', edge_cmap='Category10')
renderer = hv.renderer('bokeh')

renderer.save(chord,"t")
hv.extension('bokeh')
from bokeh.plotting import show

show(hv.render(chord))



df_json = pd.DataFrame(source_data)
            
df_links = pd.DataFrame(columns = ['source', 'target', 'value'])
            
df_links['source'] = df_json['from'].values.tolist()
df_links['target'] = df_json['to'].values.tolist()
df_links['value'] = df_json['value'].values.tolist()
            
unique_names = set(df_json['from'].unique().tolist() + df_json['to'].unique().tolist())
            
df_nodes = pd.DataFrame(columns=['name', 'group'])
df_nodes['name'] = list(unique_names)
            
## Grouping the nodes
df_nodes['group'][0:5] = 0
df_nodes['group'][5:10] = 1
df_nodes['group'][10:] = 2
            
## mapping to numerical value for the source and target node
mapper = {key: i for i, key in enumerate(list(unique_names))}
df_links['source'] = df_links['source'].map(mapper)
df_links['target'] = df_links['target'].map(mapper)
            
links = df_links
nodes = hv.Dataset(df_nodes, 'index')
            
chord = hv.Chord((links, nodes)).select(value=(1, None))
chord.opts(
          opts.Chord(cmap='Category10', edge_cmap='Category10', edge_color=dim('source').str(), 
                           labels='name', node_color=dim('group').str()))
'''

#### Clustered Heatmaps
'''
NMS=['Cleaved H3',
 'H3K27me3',
 'H3K9me3',
 'BMI-1',
 'c-Myc',
 'H3K27ac',
 'H3K36me2',
 'H4K16ac',
 'H3K27M'
 ]
df_=d0
df_=df_.append(d1)
df_=df_.append(d2)
df_=df_.append(d3)
df_=df_.append(d4)
import scipy.cluster.hierarchy as sch
for b in [0,1,2,3,4]:
        plt.figure(figsize=(10,10))
        df=df_[df_['ind']==b][NMS]
        
        
        
        cluster_th = 2
        
        X = df.corr().values
        d = sch.distance.pdist(X)
        L = sch.linkage(d, method='complete')
        ind = sch.fcluster(L, 0.5*d.max(), 'distance')
        
        columns = [df.columns.tolist()[i] for i in list(np.argsort(ind))]
        df = df.reindex(columns, axis=1)
        
        unique, counts = np.unique(ind, return_counts=True)
        counts = dict(zip(unique, counts))
        
        i = 0
        j = 0
        columns = []
        for cluster_l1 in set(sorted(ind)):
            j += counts[cluster_l1]
            sub = df[df.columns.values[i:j]]
            if counts[cluster_l1]>cluster_th:        
                X = sub.corr().values
                d = sch.distance.pdist(X)
                L = sch.linkage(d, method='complete')
                ind = sch.fcluster(L, 0.5*d.max(), 'distance')
                col = [sub.columns.tolist()[i] for i in list((np.argsort(ind)))]
                sub = sub.reindex_axis(col, axis=1)
            cols = sub.columns.tolist()
            columns.extend(cols)
            i = j
        df = df.reindex(columns, axis=1)
        
        plt.figure(figsize=(10,10))
        sns.heatmap(df.corr(),annot=True,
                cmap=plt.cm.jet,vmin=-.2,vmax=1,linewidths=.1); 
        plt.xticks(rotation=90); 
        plt.yticks(rotation=0); 
        plt.title('Ind '+str(b))
        plt.show()
'''
