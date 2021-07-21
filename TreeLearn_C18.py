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
import re
import regex
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
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
import graphviz
from sklearn.model_selection import cross_val_score
import itertools
from sklearn.tree import DecisionTreeClassifier, plot_tree
import xgboost
import shap

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
dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets/17321/"
df=pd.read_csv(dir+"C18_Norm_EpiGen_UMAP_Gated.csv")
dfmut=df


NamesAll=[
 'H3',
 'IdU',
 'MBP',
 'GFAP',
 'EZH2',
 'H3K4me3',
 'H3K79me2',
 'pHistone H2A.X [Ser139]',
 'H3K36me2',
 'Sox2',
 'SIRT1',
 'H4K16ac',
 'H2AK119ub',
 'H3K4me1',
 'H3.3',
 'H3K64ac',
 'BMI1',
 'Cmyc',
 'H4',
 'PDGFRa',
 'H4K20me3',
 'DLL3',
 'cleaved H3',
 'H3K9ac',
 'H3K27ac',
 'CD24',
 'H3K27me3',
 'H3K27M',
 'H3K9me3',
 'CD44',
 'Ki-67',
 'CXCR4',
 'pHistone H3 [S28]',
 'H1',
 'tsne0',
 'tsne1',
 'clust']

NamesTree=[
 'H3',
 'H3K4me3',
 'H3K79me2',
 'pHistone H2A.X [Ser139]',
 'H3K36me2',
 'H4K16ac',
 'H2AK119ub',
 'H3K4me1',
 'H3.3',
 'H3K64ac',
 'H4',
 'H4K20me3',
 'cleaved H3',
 'H3K9ac',
 'H3K27ac',
 'H3K27me3',
 'H3K27M',
 'H3K9me3',
 'pHistone H3 [S28]',
 'H1',
]



plt.figure(figsize=(6, 5))


dfmut=dfmut[dfmut['clust']!=-1]
mask=dfmut==-100
dfmut[mask]=0

X=dfmut[NamesAll]
Y=dfmut['clust']
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=0.33, 
                                                    random_state=42)

clf = tree.DecisionTreeClassifier(random_state=0, max_depth=2, min_samples_leaf=10)
clf = clf.fit(X_train[NamesTree], y_train)
plt.figure(figsize=(6, 5))    
tree.plot_tree(clf) 

pred=clf.predict(X_test[NamesTree])



cList=['blue','red','green','magenta','pink']
plt.figure(figsize=(10,10))
for i in range(0,2):
    mask=y_train==i    
    plt.scatter(X_train[mask]['tsne0'],X_train[mask]['tsne1'],s=2,
                c=cList[i],label="Cluster "+str(i))    
plt.legend(markerscale=5);
plt.title("C18 Trained Clusters");

cList=['blue','red','green','magenta','pink']
plt.figure(figsize=(10,10))
for i in range(0,2):
    mask=pred==i    
    plt.scatter(X_test[mask]['tsne0'],X_test[mask]['tsne1'],s=2,
                c=cList[i],label="Cluster "+str(i))    
plt.legend(markerscale=5);
plt.title("C18 Predicetd Clusters");

cList=['blue','red','green','magenta','pink']
plt.figure(figsize=(10,10))
for i in range(0,2):
    mask=y_test==i    
    plt.scatter(X_test[mask]['tsne0'],X_test[mask]['tsne1'],s=2,
                c=cList[i],label="Cluster "+str(i))    
plt.legend(markerscale=5);
plt.title("C18 Real Clusters");



print("Mean Accuracy Score %5.3f" % clf.score(X_train[NamesTree],y_train))
r = export_text(clf, feature_names=NamesTree)
#print(r)

dot_data = tree.export_graphviz(clf,
                                feature_names=NamesTree,
                                class_names=
                                ['Cluster 0','Cluster 1','Cluster 2',
                                 'Cluster 3','Cluster 4','Cluster 5','Cluster 7'],
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
r = export_text(clf, feature_names=NamesTree)
plt.rcParams.update({'font.size': 14})
from dtreeviz.trees import dtreeviz # remember to load the package
viz = dtreeviz(clf, X_train[NamesTree], y_train,
                target_name="target",
                feature_names=NamesTree,
                #class_names=True,
                fancy=True,fontname="Arial",
                title_fontsize=12,label_fontsize=16)
viz.save("C18_decision_tree.svg")




#### Descision Surfaces

# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

FeatureDict=(dict(zip(X[NamesTree].columns, clf.feature_importances_)))

qqq={k:v for (k,v) in FeatureDict.items() if v > 0}
qqq=sorted(qqq.items(), key=lambda item: item[1],reverse=True)

{print(k+" %5.3f"%v) for (k,v) in qqq if v > 0}
ql={k for (k,v) in FeatureDict.items() if v > 0}
ql=list(ql)
CombList=list(itertools.combinations(ql, 2))

# NamesTreeXGB=[
#  'H3',
#  'IdU',
#  'MBP',
#  'GFAP',
#  'EZH2',
#  'H3K4me3',
#  'H3K79me2',
#  'pHistone H2A.X _Ser139_',
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
#  'PDGFRa',
#  'H4K20me3',
#  'DLL3',
#  'cleaved H3',
#  'H3K9ac',
#  'H3K27ac',
#  'CD24',
#  'H3K27me3',
#  'H3K27M',
#  'H3K9me3',
#  'CD44',
#  'Ki-67',
#  'CXCR4',
#  'pHistone H3 _S28_',
#  'H1'
#  ]

# regex = re.compile(r"\[|\]|<", re.IGNORECASE)
# dfmutXGB=dfmut
# dfmutXGB.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) 
#                     else col for col in dfmutXGB.columns.values] 

# X=dfmutXGB[NamesTreeXGB]
# Y=dfmutXGB['clust']
# xgb_full = xgboost.DMatrix(X, label=Y)
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
# xgb_train = xgboost.DMatrix(X_train, label=y_train)
# xgb_test = xgboost.DMatrix(X_test, label=y_test)
# params = {
#     "eta": 0.002,
#     "max_depth": 3,
#     "objective": "multi:softmax",
#     "subsample": 0.5,
#     "num_class": 2
# }
# model_train = xgboost.train(params, xgb_train, 10000, evals = [(xgb_test, "test")], verbose_eval=1000)
# model = xgboost.train(params, xgb_full, 5000, evals = [(xgb_full, "test")], verbose_eval=1000)
# shap_values = shap.TreeExplainer(model).shap_values(X)

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