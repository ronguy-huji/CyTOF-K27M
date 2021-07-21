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



#df=pd.read_csv("control.csv")
#dfmut=pd.read_csv("mutant.csv")
dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets/17321/"


aaa=pd.read_csv(dir+"C16_C17_Norm_Clust_Full_Panel.csv")




NamesAll=['H3',
 'IdU',
 'MBP',
 'H3K36me3',
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
 'pHistone H3 [S28]',
 'H1']



plt.figure(figsize=(6, 5))



for NN in NamesAll:
    BoxVar=NN
    plt.figure(figsize=(6, 5))    
    ax = sns.boxplot(x="clust", y=NN, data=aaa,showfliers=False)
    plt.title(NN+" C16/C17 Full Panel")
    
    
    


