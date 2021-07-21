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
import sklearn.metrics
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
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
import graphviz
from sklearn.model_selection import cross_val_score
import itertools
from sklearn.tree import DecisionTreeClassifier, plot_tree

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
#df=pd.read_csv(dir+"C13_Norm.csv")
dfmut=pd.read_csv(dir+"C14_UMAP.csv")



Names=['H2Aub',
# 'H2B',
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
 'H4K16ac',
 'cleaved H3',
 'yH2A.X',
 #'H2A',
 'pH3[S28]']



#df=df[Names]
dfmut=dfmut[dfmut['clust']!=-1]
dfmut=dfmut[dfmut['clust']!=2]
plt.figure(figsize=(6, 5))




'''
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



lab=dbscan_plot(X_2d,eps=0.181,min_samples=115)
'''

'''
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
NamestSNE=['H2Aub',
 'H2B',
 'H3',
 'H3.3',
 'H3K27M',
 'H3K27ac',
 'H3K27me3',
 'H3K36me2',
 'H3K36me3',
 'H3K4me1',
 'H3K4me3',
 'H3K64ac',
 'H3K9ac',
 'H3K9me3',
 'H4',
 'H4K16ac',
 'cleaved H3',
 #'H2A',
 'pH3[S28]',
 'yH2A.X',
 'umap0',
 'umap1']

Names=['H2Aub',
# 'H2B',
# 'H3',
# 'H3.3',
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
# 'H4',
 'H4K16ac',
 'cleaved H3',
 'yH2A.X',
 #'H2A',
 'pH3[S28]']

X=dfmut[NamestSNE]
Y=dfmut['clust']
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=0.33, 
                                                    random_state=40)

clf = tree.DecisionTreeClassifier(random_state=0, max_depth=3, min_samples_leaf=50)
clf = clf.fit(X_train[Names], y_train)
plt.figure(figsize=(6, 5))    
tree.plot_tree(clf) 

pred=clf.predict(X_test[Names])

plt.figure(figsize=(6, 5))    
mask=y_train==0
plt.scatter(X_train[mask]['umap0'],X_train[mask]['umap1'],color='darkorange',s=2)
mask=y_train==1
plt.scatter(X_train[mask]['umap0'],X_train[mask]['umap1'],color='dimgray',s=2)

plt.title('Trained')

plt.figure(figsize=(6, 5)) 
mask=pred==0   
plt.scatter(X_test[mask]['umap0'],X_test[mask]['umap1'],color='darkorange',s=2)
mask=pred==1
plt.scatter(X_test[mask]['umap0'],X_test[mask]['umap1'],color='dimgray',s=2)  
plt.title('Predicted')




disp = sklearn.metrics.plot_confusion_matrix(clf, X_test[Names], y_test,
                                 display_labels=['H3K27M High','H3K27M Low'],
                                 cmap=plt.cm.Blues,
                                 normalize='true')

print("Mean Accuracy Score %5.3f" % clf.score(X_train[Names],y_train))
r = export_text(clf, feature_names=Names)
#print(r)

dot_data = tree.export_graphviz(clf,
                                feature_names=Names,
                                class_names=['Cluster 0','Cluster 1'],
                                filled=True, rounded=True,
                                leaves_parallel=True,)
                       

'''tree.export_graphviz(clf, out_file=None, 
                      feature_names=Names,  
                      class_names=['Cluster 0','Cluster 1','Cluster 2'],  
                      filled=True, rounded=True,  
                      special_characters=True,
                      proportion = False, label='none'
                      )  
'''
graph = graphviz.Source(dot_data)  
graph.render("Clust")
r = export_text(clf, feature_names=Names)
plt.rcParams.update({'font.size': 14})
from dtreeviz.trees import dtreeviz # remember to load the package
viz = dtreeviz(clf, X_train[Names], y_train,
                target_name="Cluster",
                feature_names=Names,
                class_names=['Cluster 0','Cluster 1'],
                fancy=True,fontname="Arial",
                title_fontsize=20,label_fontsize=16,
                colors={'classes':[0,0,['darkorange','dimgray']]})
viz.save("decision_tree.svg")




#### Descision Surfaces

# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

FeatureDict=(dict(zip(X[Names].columns, clf.feature_importances_)))

qqq={k:v for (k,v) in FeatureDict.items() if v > 0}
qqq=sorted(qqq.items(), key=lambda item: item[1],reverse=True)

{print(k+" %5.3f"%v) for (k,v) in qqq if v > 0}
ql={k for (k,v) in FeatureDict.items() if v > 0}
ql=list(ql)
CombList=list(itertools.combinations(ql, 2))


'''

plt.figure(figsize=(10, 10))  
for pairidx, pair in enumerate(CombList):
    # We only take the two corresponding features
    X = dfmut[list(pair)].values
    y = dfmut['clust'].values

    # Train
    clf = DecisionTreeClassifier().fit(X, y)

    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel(pair[0])
    plt.ylabel(pair[1])

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=str(i),
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree using paired features")
#plt.legend(loc='lower right', borderpad=0, handletextpad=0)
#plt.axis("tight")
'''


#####

features=[#'H2AK119ub',
# 'H2B',
# 'H3',
# 'H3.3',
 'H3K27ac',
 'H3K27M',
# 'H3K27me3',
# 'H3K36me2',
# 'H3K36me3',
# 'H3K4me1',
# 'H3K4me3',
# 'H3K64ac',
# 'H3K9ac',
 'H3K9me3',
# 'H4',
 'H4K16ac',
 'cleaved H3',
 'yH2A.X',
 #'H2A',
# 'pHistone H3 [S28]'
]
'''
random.shuffle(features)

ddf=dfmut.sample(frac=1).sort_values('clust',ascending=True)
#ddf=ddf[ddf['clust'].isin([0,1])]
plt.figure(figsize=(15,20))
ax=pd.plotting.radviz(ddf[features+['clust']],class_column='clust',
                     color=['red','blue','green','magenta'],
                     s=150,alpha=0.5)
#for text in l.get_texts():
#    text.set_fontsize(10)
#plt.xticks(fontsize=10)

ax.legend( loc='center left', bbox_to_anchor=(-.1, 1),
               fontsize=50, fancybox=True, ncol=4 ,markerscale=2)
ax.set_title( 'C14', loc='right', fontsize=50,color='black' )
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
SP=['left','right','top','bottom']
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

for p in ax.patches[1:]:
    p.set_visible(True)
    p.set_color('black')

ax.patches[0].set_color('green')
ax.patches[0].set_alpha(0.1)

for t in ax.texts:
    t.set_color('black')
    t.set_fontsize(50)
    
plt.show()
'''

mask0=dfmut['clust']==0

mask1=dfmut['clust']==1
fig, ax = plt.subplots(figsize=(15,10), dpi= 80)
sns.kdeplot(dfmut[mask0]['H3K9ac'],color='red',label='H3K9ac'); 
sns.kdeplot(dfmut[mask0]['H3K64ac'],color='blue',label='H3K9ac');
plt.xlabel("Value")
plt.title("C14 K27M High Cluster")
plt.legend();
plt.show()

fig, ax = plt.subplots(figsize=(15,10), dpi= 80)
sns.kdeplot(dfmut[mask1]['H3K9ac'],color='red',label='H3K9ac'); 
sns.kdeplot(dfmut[mask1]['H3K64ac'],color='blue',label='H3K9ac');
plt.xlabel("Value")
plt.title("C14 K27M Low Cluster")
plt.legend();
plt.show()

