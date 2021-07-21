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


#### My Waterfall Plot

from shap.utils import safe_isinstance, format_value
from shap.plots._labels import labels
from shap.plots import colors
import matplotlib.pyplot as pl
import matplotlib

def wfall(shap_values, max_display=10, show=True):
    """ Plots an explantion of a single prediction as a waterfall plot.
    The SHAP value of a feature represents the impact of the evidence provided by that feature on the model's
    output. The waterfall plot is designed to visually display how the SHAP values (evidence) of each feature
    move the model output from our prior expectation under the background data distribution, to the final model
    prediction given the evidence of all the features. Features are sorted by the magnitude of their SHAP values
    with the smallest magnitude features grouped together at the bottom of the plot when the number of features
    in the models exceeds the max_display parameter.
    
    Parameters
    ----------
    shap_values : Explanation
        A one-dimensional Explanation object that contains the feature values and SHAP values to plot.
    max_display : str
        The maximum number of features to plot.
    show : bool
        Whether matplotlib.pyplot.show() is called before returning. Setting this to False allows the plot
        to be customized further after it has been created.
    """
    dark_o= mpl.colors.to_rgb('dimgray')
    dim_g= mpl.colors.to_rgb('darkorange')

    base_values = shap_values.base_values
    
    features = shap_values.data
    feature_names = shap_values.feature_names
    lower_bounds = getattr(shap_values, "lower_bounds", None)
    upper_bounds = getattr(shap_values, "upper_bounds", None)
    values = shap_values.values

    # make sure we only have a single output to explain
    if (type(base_values) == np.ndarray and len(base_values) > 0) or type(base_values) == list:
        raise Exception("waterfall_plot requires a scalar base_values of the model output as the first " \
                        "parameter, but you have passed an array as the first parameter! " \
                        "Try shap.waterfall_plot(explainer.base_values[0], values[0], X[0]) or " \
                        "for multi-output models try " \
                        "shap.waterfall_plot(explainer.base_values[0], values[0][0], X[0]).")

    # make sure we only have a single explanation to plot
    if len(values.shape) == 2:
        raise Exception("The waterfall_plot can currently only plot a single explanation but a matrix of explanations was passed!")
    
    # unwrap pandas series
    if safe_isinstance(features, "pandas.core.series.Series"):
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values

    # fallback feature names
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(values))])
    
    # init variables we use for tracking the plot locations
    num_features = min(max_display, len(values))
    row_height = 0.5
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(values))
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    loc = base_values + values.sum()
    yticklabels = ["" for i in range(num_features + 1)]
    
    # size the plot based on how many features we are plotting
    pl.gcf().set_size_inches(8, num_features * row_height + 1.5)

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features == len(values):
        num_individual = num_features
    else:
        num_individual = num_features - 1

    # compute the locations of the individual features and plot the dashed connecting lines
    for i in range(num_individual):
        sval = values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            pl.plot([loc, loc], [rng[i] -1 - 0.4, rng[i] + 0.4], color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
        if features is None:
            yticklabels[rng[i]] = feature_names[order[i]]
        else:
            yticklabels[rng[i]] = format_value(features[order[i]], "%0.03f") + " = " + feature_names[order[i]] 
    
    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(values):
        yticklabels[0] = "%d other features" % (len(values) - num_features + 1)
        remaining_impact = base_values - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
            c = dim_g  #colors.red_rgb
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)
            c = dark_o #colors.blue_rgb

    points = pos_lefts + list(np.array(pos_lefts) + np.array(pos_widths)) + neg_lefts + list(np.array(neg_lefts) + np.array(neg_widths))
    dataw = np.max(points) - np.min(points)
    
    # draw invisible bars just for sizing the axes
    label_padding = np.array([0.1*dataw if w < 1 else 0 for w in pos_widths])
    pl.barh(pos_inds, np.array(pos_widths) + label_padding + 0.02*dataw, left=np.array(pos_lefts) - 0.01*dataw, color=colors.red_rgb, alpha=0)
    label_padding = np.array([-0.1*dataw  if -w < 1 else 0 for w in neg_widths])
    pl.barh(neg_inds, np.array(neg_widths) + label_padding - 0.02*dataw, left=np.array(neg_lefts) + 0.01*dataw, color=colors.blue_rgb, alpha=0)
    
    # define variable we need for plotting the arrows
    head_length = 0.08
    bar_width = 0.8
    xlen = pl.xlim()[1] - pl.xlim()[0]
    fig = pl.gcf()
    ax = pl.gca()
    xticks = ax.get_xticks()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    bbox_to_xscale = xlen/width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()
    
    # draw the positive arrows
    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = pl.arrow(
            pos_lefts[i], pos_inds[i], max(dist-hl_scaled, 0.000001), 0,
            head_length=min(dist, hl_scaled),
            color=dim_g, width=bar_width,
            head_width=bar_width
        )
        
        if pos_low is not None and i < len(pos_low):
            pl.errorbar(
                pos_lefts[i] + pos_widths[i], pos_inds[i], 
                xerr=np.array([[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]),
                ecolor=dim_g
            )

        txt_obj = pl.text(
            pos_lefts[i] + 0.5*dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=12
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)
        
        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width: 
            txt_obj.remove()
            
            txt_obj = pl.text(
                pos_lefts[i] + (5/72)*bbox_to_xscale + dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
                horizontalalignment='left', verticalalignment='center', color=dim_g,
                fontsize=12
            )
    
    # draw the negative arrows
    for i in range(len(neg_inds)):
        dist = neg_widths[i]
        
        arrow_obj = pl.arrow(
            neg_lefts[i], neg_inds[i], -max(-dist-hl_scaled, 0.000001), 0,
            head_length=min(-dist, hl_scaled),
            color=dark_o, width=bar_width,
            head_width=bar_width
        )

        if neg_low is not None and i < len(neg_low):
            pl.errorbar(
                neg_lefts[i] + neg_widths[i], neg_inds[i], 
                xerr=np.array([[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]),
                ecolor=dark_o
            )
        
        txt_obj = pl.text(
            neg_lefts[i] + 0.5*dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=12
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)
        
        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width: 
            txt_obj.remove()
            
            txt_obj = pl.text(
                neg_lefts[i] - (5/72)*bbox_to_xscale + dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
                horizontalalignment='right', verticalalignment='center', color=dark_o,
                fontsize=12
            )

    # draw the y-ticks twice, once in gray and then again with just the feature names in black
    ytick_pos = list(range(num_features)) + list(np.arange(num_features)+1e-8) # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    pl.yticks(ytick_pos, yticklabels[:-1] + [l.split('=')[-1] for l in yticklabels[:-1]], fontsize=13)
    
    # put horizontal lines for each feature row
    for i in range(num_features):
        pl.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
    
    # mark the prior expected value and the model prediction
    pl.axvline(base_values, 0, 1/num_features, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
    fx = base_values + values.sum()
    pl.axvline(fx, 0, 1, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
    
    # clean up the main axis
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    ax.tick_params(labelsize=13)
    #pl.xlabel("\nModel output", fontsize=12)

    # draw the E[f(X)] tick mark
    xmin,xmax = ax.get_xlim()
    ax2=ax.twiny()
    ax2.set_xlim(xmin,xmax)
    ax2.set_xticks([base_values, base_values+1e-8]) # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax2.set_xticklabels(["\n$E[f(X)]$","\n$ = "+format_value(base_values, "%0.03f")+"$"], fontsize=12, ha="left")
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # draw the f(x) tick mark
    ax3=ax2.twiny()
    ax3.set_xlim(xmin,xmax)
    ax3.set_xticks([base_values + values.sum(), base_values + values.sum() + 1e-8]) # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax3.set_xticklabels(["$f(x)$","$ = "+format_value(fx, "%0.03f")+"$"], fontsize=12, ha="left")
    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-10/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform() + matplotlib.transforms.ScaledTranslation(12/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_color("#999999")
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    # adjust the position of the E[f(X)] = x.xx label
    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-20/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform() + matplotlib.transforms.ScaledTranslation(22/72., -1/72., fig.dpi_scale_trans))
    
    tick_labels[1].set_color("#999999")

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color("#999999")
    
    if show:
        pl.show()






#df=pd.read_csv("control.csv")
#dfmut=pd.read_csv("mutant.csv")
dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets/C13C14/"
#df=pd.read_csv(dir+"C13_Norm.csv")
dfmut=pd.read_csv(dir+"C14_UMAP.csv")

dfmut['clust'].replace({0:1,1:0}, inplace=True)

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



### Create Orange/Gray Colormap
from matplotlib import cm
import matplotlib as mpl
RGB_val = 1#255

color01= mpl.colors.to_rgb('dimgray')
color04= mpl.colors.to_rgb('darkorange')
Colors = [color01, color04]

# Creating a blue red palette transition for graphics
Colors= [(R/RGB_val,G/RGB_val,B/RGB_val) for idx, (R,G,B) in enumerate(Colors)]
n = 128

# Start of the creation of the gradient
Color01= mpl.colors.ListedColormap(Colors[0], name='Color01', N=None)
Color04= mpl.colors.ListedColormap(Colors[1], name='Color04', N=None)
top = cm.get_cmap(Color01,128)
bottom = cm.get_cmap(Color04,128)
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))

mymin0 = newcolors[0][0]
mymin1 = newcolors[0][1]
mymin2 = newcolors[0][2]
mymin3 = newcolors[0][3]
mymax0 = newcolors[255][0]
mymax1 = newcolors[255][1]
mymax2 = newcolors[255][2]
mymax3 = newcolors[255][3]

mymid=1


GradientOrangeWhite= [np.linspace(mymin0, mymid,  n),
                   np.linspace(mymin1, mymid,  n),
                   np.linspace(mymin2, mymid,  n),
                   np.linspace(mymin3, mymid,  n)]

GradientWhiteGray= [np.linspace(mymid, mymax0,  n),
                   np.linspace(mymid, mymax1,  n),
                   np.linspace(mymid, mymax2,  n),
                   np.linspace(mymid, mymax3,  n)]
GradientOrangeGray=np.concatenate((GradientOrangeWhite, GradientWhiteGray),axis=1)
GradientOrangeGray_res =np.transpose(GradientOrangeGray)

# End of the creation of the gradient

newcmp = mpl.colors.ListedColormap(GradientOrangeGray_res, name='OrangeGray')



####

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
plt.scatter(X_train[mask]['umap0'],X_train[mask]['umap1'],color='dimgray',s=2)
mask=y_train==1
plt.scatter(X_train[mask]['umap0'],X_train[mask]['umap1'],color='darkorange',s=2)

plt.title('Trained')

plt.figure(figsize=(6, 5)) 
mask=pred==0   
plt.scatter(X_test[mask]['umap0'],X_test[mask]['umap1'],color='dimgray',s=2)
mask=pred==1
plt.scatter(X_test[mask]['umap0'],X_test[mask]['umap1'],color='darkorange',s=2)  
plt.title('Predicted')




disp = sklearn.metrics.plot_confusion_matrix(clf, X_test[Names], y_test,
                                 display_labels=['H3K27M High','H3K27M Low'],
                                 cmap=plt.cm.Blues,
                                 normalize='true')

print("Mean Accuracy Score %5.3f" % clf.score(X_train[Names],y_train))

print("CLF Accuracy Score %5.3f" % sklearn.metrics.accuracy_score(y_test,pred))

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
                colors={'classes':[0,0,['dimgray','darkorange']]})
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



import xgboost as xgb
import re
import regex
import shap

NMS=[
     'H2Aub',
     'H3K27ac',
     'H3K27M',
#     'H3K27me3',
#     'H3K36me2',
#     'H3K36me3',
     'H3K4me1',
#     'H3K4me3',
#     'H3K64ac',
#     'H3K9ac',
     'H3K9me3',
     'H4K16ac',
     'cleaved H3',
     'yH2A.X',
     'pH3_S28_'
    ]
    
dfXGB=dfmut.copy()


regex = re.compile(r"\[|\]|<", re.IGNORECASE)
dfXGB.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<')))
                     else col for col in dfXGB.columns.values]

X=dfXGB
Y=dfXGB['clust']


X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=0.33, 
                                                    random_state=40)


xgb_binary_model = xgb.XGBRegressor(objective='reg:logistic',random_state=42,)
#                                    feature_names=NMS)
xgb_binary_model.fit(X_train[NMS], y_train)

pred_prob=xgb_binary_model.predict(X_test[NMS],output_margin=False)
pred = np.where(pred_prob>0.5,1,0)


print("XGB Accuracy Score %5.3f" % sklearn.metrics.accuracy_score(y_test,pred))


explainer = shap.TreeExplainer(xgb_binary_model,data=X_train[NMS])
xgb_binary_shap_values = explainer.shap_values(X_train[NMS])

shap_values=xgb_binary_shap_values

 


def xgb_shap_transform_scale(original_shap_values, Y_pred, which):
    from scipy.special import expit #Importing the logit function for the base value transformation
    untransformed_base_value = original_shap_values.base_values[-1]
    base_value = expit(original_shap_values.base_values ) # = 1 / (1+ np.exp(-untransformed_base_value))
    
    original_explanation_distance = np.sum(shap_values.values, axis=1)[which]
    
    distance_to_explain = abs(Y_pred[which] - base_value[which])
    distance_coefficient = np.abs(original_explanation_distance / distance_to_explain)
    shap_values_transformed = original_shap_values / distance_coefficient
    shap_values_transformed.base_values = base_value
    shap_values_transformed.data = shap_values.data

    return shap_values_transformed    

explainer = shap.Explainer(xgb_binary_model)
shap_values = explainer(X_train[NMS])
shap.plots.waterfall(shap_values[0])

p = 0.5  # Probability 0.4
new_base_value = np.log(p / (1 - p))  # the logit function
expected_value = explainer.expected_value
shap_v = explainer(X_train[NMS])[0]
shap.decision_plot(expected_value, shap_values[0:100].values, 
                   X_train[NMS],link='logit',feature_order='hclust',
                   new_base_value=new_base_value, plot_color=newcmp)


import warnings
y_pred = xgb_binary_model.predict(X_train[NMS])
T = X_train[NMS][y_pred >= 0.90]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sh = explainer.shap_values(T)[0:500]
shap.decision_plot(expected_value, sh, T, feature_order='hclust', link='logit',
                   new_base_value=new_base_value,plot_color=newcmp)


y_pred = xgb_binary_model.predict(X_train[NMS])
T = X_train[NMS][y_pred <= 0.10]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sh = explainer.shap_values(T)[0:500]
shap.decision_plot(expected_value, sh, T, feature_order='hclust', link='logit',
                   new_base_value=new_base_value,plot_color=newcmp)


# visualize the first prediction's explanation
obs = 0
Y_pred = xgb_binary_model.predict(X_train[NMS])
print("The prediction is ", Y_pred[obs])
shap_values_transformed = xgb_shap_transform_scale(shap_values, Y_pred, obs)
wfall(shap_values_transformed[obs])

obs = 3
Y_pred = xgb_binary_model.predict(X_train[NMS])
print("The prediction is ", Y_pred[obs])
shap_values_transformed = xgb_shap_transform_scale(shap_values, Y_pred, obs)
wfall(shap_values_transformed[obs])

XT = X_train[1:1000].sort_values(by='clust')
shap_values = explainer(XT[NMS])
a=np.arange(0,998)
shap.plots.heatmap(shap_values[1:1000],cmap=newcmp,instance_order=a)





explainer = shap.TreeExplainer(xgb_binary_model)
expected_value = explainer.expected_value
print("The expected value is ", expected_value)
shap_values = explainer.shap_values(X_train[NMS])


from sklearn.inspection import permutation_importance

fig, ax = plt.subplots(figsize=(15,10), dpi= 80)
shap.summary_plot(shap_values, X_train[NMS], plot_type="bar")
plt.show()

fig, ax = plt.subplots(figsize=(15,10), dpi= 80)
sorted_idx = xgb_binary_model.feature_importances_.argsort()
plt.barh(np.asanyarray(NMS)[sorted_idx], xgb_binary_model.feature_importances_[sorted_idx])
plt.show()
for nam,imp in zip(np.asanyarray(NMS)[sorted_idx],xgb_binary_model.feature_importances_[sorted_idx]):
               print(nam,imp)


fig, ax = plt.subplots(figsize=(15,10), dpi= 80)
perm_importance = permutation_importance(xgb_binary_model, X_test[NMS], y_test)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(np.asanyarray(NMS)[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.show()


def correlation_heatmap(train):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(15,15))
    sns.clustermap(correlations, vmax=1.0, vmin=-1., center=0, fmt='.2f', cmap="jet",
                linewidths=.5, annot=True, cbar_kws={"shrink": .70}
                )
    plt.show();
    
correlation_heatmap(X_train[np.asanyarray(NMS)[sorted_idx]])



plt.figure(figsize=(6, 5))    
mask=y_train==0
plt.scatter(X_train[mask]['umap0'],X_train[mask]['umap1'],color='dimgray',s=2)
mask=y_train==1
plt.scatter(X_train[mask]['umap0'],X_train[mask]['umap1'],color='darkorange',s=2)

plt.title('Trained')

plt.figure(figsize=(6, 5)) 
mask=pred==0   
plt.scatter(X_test[mask]['umap0'],X_test[mask]['umap1'],color='dimgray',s=2)
mask=pred==1
plt.scatter(X_test[mask]['umap0'],X_test[mask]['umap1'],color='darkorange',s=2)  
plt.title('Predicted')

cm=sklearn.metrics.confusion_matrix(y_test,pred,normalize="true")

disp = sklearn.metrics.ConfusionMatrixDisplay(cm,display_labels=['H3K27M High','H3K27M Low'],
                                              )
disp.plot(cmap=plt.cm.Blues)



def ABS_SHAP(df_shap,df):
    #import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'darkorange','dimgray')
    corr_df['SGN'] = np.where(corr_df['Corr']>0,1,-1)
    # Plot it
    
    shap_abs = np.abs(shap_v)

    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
#    print(shap_abs.mean())
#    print(shap_v.mean())
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    print(k2)
    k2 = k2.assign(SV=k2['SHAP_abs']*k2['SGN'])
    k2 = k2.sort_values(by='SV',ascending = True)
    colorlist = k2['Sign']
#    print(colorlist)
#    plt.figure(figsize=(15, 10)) 
    ax = k2.plot.barh(x='Variable',y='SV',color = colorlist, figsize=(10,8),legend=False)
    ax.set_xlabel("SHAP Value")
    ax.set_xlim(-2.5,2.5)
    
ABS_SHAP(shap_values,X_train[NMS]) 


g=sns.pairplot(X[NMS+['clust']],hue='clust',corner=True,
              diag_kind="kde",diag_kws=dict(fill=True, common_norm=False),
              palette={0:'dimgray',1:'darkorange'})
g._legend.set_title('Cluster')
new_labels = ['H3K27M Low', 'H3K27M High']
plt.legend(fontsize=20,
            facecolor='White', framealpha=1,frameon=True)
for t, l in zip(g._legend.texts, new_labels): 
    t.set_text(l)
    t.set_fontsize(20)
plt.show()



#######

#Rewrite RadViz function
    
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
    colors=['darkorange','dimgray']
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

    ax.add_patch(patches.Circle((0.0, 0.0), radius=.50, facecolor="none",edgecolor='black'))
    s=s*.5
    for xy, name in zip(s, df.columns):
        rad=0.05*0.5
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

from pandas.io.formats.printing import pprint_thing
features=[
    'H3K4me1', 
    'H3K27ac', 
    'H3K27M', 
    'H4K16ac',
#    'H2Aub',
#    'yH2A.X',
    'pH3_S28_',
    'H3K9me3',
    'cleaved H3'
    ]
plt.figure(figsize=(10,10))
ax,tp=radviz2(X[features+['clust']],class_column='clust',
                     color=['darkorange','dimgray'],
                     s=20,alpha=0.3)


x0=np.asarray(tp[0][0]); 
y0=np.asarray(tp[0][1]); 
x1=np.asarray(tp[1][0]); 
y1=np.asarray(tp[1][1]); 

xy = np.vstack([x0,y0])

pts0 = xy.mean(axis=1)#xy.T[np.argmax(density)]

xy = np.vstack([x1,y1])

pts1 = xy.mean(axis=1)#xy.T[np.argmax(density)]





ax.legend( loc='center left', bbox_to_anchor=(1.25, 1),
               fontsize=30, fancybox=True, ncol=1 ,markerscale=2)
#ax.set_title( ExpName, loc='right', fontsize=30,color='black' )
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
ax.scatter(pts1[0],pts1[1],color='darkorange',s=300,ec='k',linewidth=4,alpha=1,zorder=100);
ax.scatter(pts0[0],pts0[1],color='dimgray',s=300,ec='k',linewidth=4,alpha=1,zorder=105);
#ax.scatter(0,0,color='black',s=200,ec='k',linewidth=3,alpha=1,zorder=115);
plt.show()


'''

# Decision region for feature 3 = 1.5
value = 0
# Plot training sample with feature 3 = 1.5 +/- 0.75
width = 1.0
ffv={
     0:value,
#     1:value,
     2:value,
     3:value,
     4:value,
     5:value,
#     6:value,
     7:value,
     8:value,
     9:value
     }
ffr={
     0:width,
#     1:width,
     2:width,
     3:width,
     4:width,
     5:width,
#     6:width,
     7:width,
     8:width,
     9:width
     }

XT=np.asarray(X_train[NMS])
YT=np.asarray(y_train)
xgb_binary_model = xgb.XGBRegressor(objective='reg:logistic',random_state=42,
                                    feature_names=NMS)
xgb_binary_model.fit(XT,YT)

from mlxtend.plotting import plot_decision_regions
fig = plot_decision_regions(X=XT,                         
                            y=YT, feature_index=[1,6],
                            filler_feature_ranges=ffr,
                            filler_feature_values=ffv,
                            clf=xgb_binary_model, legend=2)

'''
