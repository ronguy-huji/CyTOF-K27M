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


dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets/Normalized/"
C15=pd.read_csv(dir+"C15_Norm.csv")
C16=pd.read_csv(dir+"C16_Norm.csv")

C05=pd.read_csv(dir+"C05_Norm.csv")
C07=pd.read_csv(dir+"C07_Norm.csv")

C06=pd.read_csv(dir+"C06_Norm.csv")
C08=pd.read_csv(dir+"C08_Norm.csv")

Names=[
# 'H2A',
# 'H2B',
# 'H3',
# 'H3.3',
 'H3K27M',
 'H3K27ac',
 'H4K16ac',

 'H3K27me3',
 'H3K36me2',
 'H3K36me3',
 'H3K4me1',
 'H3K4me3',
 'H3K64ac',
 'H3K9ac',
 'H3K9me3',
# 'H4',
 'cleaved H3',
 'pH3[S28]',
 'H2Aub',

]

C05m=C05[Names].mean().to_frame()
C06m=C06[Names].mean().to_frame()
C07m=C07[Names].mean().to_frame()
C08m=C08[Names].mean().to_frame()
C15m=C15[Names].mean().to_frame()
C16m=C16[Names].mean().to_frame()

M=[C15m,C06m,C05m,C16m,C08m,C07m]



Mat=np.zeros((14,6))

for j,MM in enumerate(M):
    for i in range(0,13):
        Mat[i,j]=MM.iloc[i][0]
        print(i,MM.iloc[i])
        
plt.figure(figsize=(10,6))        
sns.heatmap(Mat,yticklabels=Names,xticklabels=['C15','C06','C05','C16','C08','C07'],
            cmap=plt.cm.jet,vmin=-1.2,vmax=1.2)        
plt.yticks(rotation=0) 