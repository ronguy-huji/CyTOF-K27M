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
dir="/Users/ronguy/Dropbox/Work/CyTOF/Datasets_Normalized/"

F1="C07"
F2="C06"

#aaa=pd.read_csv(dir+"C05-C07_Norm.csv")

df=pd.read_csv(dir+F1+"-Norm-Nada.csv")
dfmut=pd.read_csv(dir+F2+"-Norm-Nada.csv")

#df=aaa[aaa['Type']=="Wild"]
#dfmut=aaa[aaa['Type']=="Mutant"]
df=df.assign(Type='Wild')
dfmut=dfmut.assign(Type='Mutant')

aaa=df
aaa=aaa.append(dfmut)

Names=['H2AK119ub',
 'H2B',
 'H3',
 'H3.3',
# 'H3K27ac',
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
 #'H2A',
 'pHistone H3 [S28]']



df=df[Names]
dfmut=dfmut[Names]

plt.figure(figsize=(6, 5))





Names_All=['H2AK119ub',
# 'H3K27ac',
 'H3K27me3',
 'H3K36me2',
 'H3K36me3',
 'H3K4me1',
 'H3K4me3',
 'H3K64ac',
 'H3K9ac',
 'H3K9me3',
 'H4K16ac',
 'cleaved H3',
 #'H2A',
 'pHistone H3 [S28]']
Open_Chromatin=[
     'H3K4me1', 'H3K4me3', 'H4K16ac', 'H3K9ac', 'H3K64ac'
     ]
Closed_Chromatin=['H3K9me3', 'H3K27me3', 'H3K36me2', 'H2AK119ub', 'H3K9ac']


### Heatmaps for wild 

#sns.heatmap(dfmut[Names].corr()-df[Names].corr(),annot=True,cmap=plt.cm.jet,vmin=0,vmax=1,linewidths=.1);
plt.figure(figsize=(10, 10))
g=sns.clustermap(df[Names_All].corr(),
                 annot=True,
                 cmap=plt.cm.jet,
                 vmin=0,vmax=1,linewidths=.1,
                 cbar_pos=(1, 0.5, 0.05, 0.18));
plt.xticks(rotation=90); 
plt.yticks(rotation=0)
g.ax_col_dendrogram.set_visible(False)
#g.ax_col_dendrogram.set_xlim([0,0])
#g.ax_cbar.set_position((0.8, .2, .03, .4))
plt.title(F1+" All"); 
plt.savefig("Plots/"+F1+"_All.png",dpi=100,bbox_inches = 'tight')
plt.show()




plt.figure(figsize=(10, 10))
g=sns.clustermap(df[Open_Chromatin].corr(),
                 annot=True,
                 cmap=plt.cm.jet,
                 vmin=0,vmax=1,linewidths=.1,
                 cbar_pos=(1, 0.5, 0.05, 0.18));
plt.xticks(rotation=90); 
plt.yticks(rotation=0)
g.ax_col_dendrogram.set_visible(False)
#g.ax_col_dendrogram.set_xlim([0,0])
#g.ax_cbar.set_position((0.8, .2, .03, .4))
plt.title(F1+" Open Chromatin");
plt.savefig("Plots/"+F1+"_Open.png",dpi=100,bbox_inches = 'tight') 
plt.show()


plt.figure(figsize=(10, 10))
g=sns.clustermap(df[Closed_Chromatin].corr(),
                 annot=True,
                 cmap=plt.cm.jet,
                 vmin=0,vmax=1,linewidths=.1,
                 cbar_pos=(1, 0.5, 0.05, 0.18));
plt.xticks(rotation=90); 
plt.yticks(rotation=0)
g.ax_col_dendrogram.set_visible(False)
#g.ax_col_dendrogram.set_xlim([0,0])
#g.ax_cbar.set_position((0.8, .2, .03, .4))
plt.title(F1+" Closed Chromatin"); 
plt.savefig("Plots/"+F1+"_Closed.png",dpi=100,bbox_inches = 'tight')
plt.show()
#### Heatmaps for Mutant

#sns.heatmap(dfmut[Names].corr()-df[Names].corr(),annot=True,cmap=plt.cm.jet,vmin=0,vmax=1,linewidths=.1);
plt.figure(figsize=(10, 10))
g=sns.clustermap(dfmut[Names_All].corr(),
                 annot=True,
                 cmap=plt.cm.jet,
                 vmin=0,vmax=1,linewidths=.1,
                 cbar_pos=(1, 0.5, 0.05, 0.18));
plt.xticks(rotation=90); 
plt.yticks(rotation=0)
g.ax_col_dendrogram.set_visible(False)
#g.ax_col_dendrogram.set_xlim([0,0])
#g.ax_cbar.set_position((0.8, .2, .03, .4))
plt.title(F2+" All"); 
plt.savefig("Plots/"+F2+"_All.png",dpi=100,bbox_inches = 'tight')
plt.show()




plt.figure(figsize=(10, 10))
g=sns.clustermap(dfmut[Open_Chromatin].corr(),
                 annot=True,
                 cmap=plt.cm.jet,
                 vmin=0,vmax=1,linewidths=.1,
                 cbar_pos=(1, 0.5, 0.05, 0.18));
plt.xticks(rotation=90); 
plt.yticks(rotation=0)
g.ax_col_dendrogram.set_visible(False)
#g.ax_col_dendrogram.set_xlim([0,0])
#g.ax_cbar.set_position((0.8, .2, .03, .4))
plt.title(F2+" Open Chromatin");
plt.savefig("Plots/"+F2+"_Open.png",dpi=100,bbox_inches = 'tight') 
plt.show()


plt.figure(figsize=(10, 10))
g=sns.clustermap(dfmut[Closed_Chromatin].corr(),
                 annot=True,
                 cmap=plt.cm.jet,
                 vmin=0,vmax=1,linewidths=.1,
                 cbar_pos=(1, 0.5, 0.05, 0.18));
plt.xticks(rotation=90); 
plt.yticks(rotation=0)
g.ax_col_dendrogram.set_visible(False)
#g.ax_col_dendrogram.set_xlim([0,0])
#g.ax_cbar.set_position((0.8, .2, .03, .4))
plt.title(F2+" Closed Chromatin"); 
plt.savefig("Plots/"+F2+"_Closed.png",dpi=100,bbox_inches = 'tight')
plt.show()
####
#### Delta correlations 


plt.figure(figsize=(10, 10))
g=sns.clustermap(dfmut[Names_All].corr()-df[Names_All].corr(),
                 annot=True,
                 cmap=plt.cm.jet,
                 vmin=0,vmax=1,linewidths=.1,
                 cbar_pos=(1, 0.5, 0.05, 0.18));
plt.xticks(rotation=90); 
plt.yticks(rotation=0)
g.ax_col_dendrogram.set_visible(False)
#g.ax_col_dendrogram.set_xlim([0,0])
#g.ax_cbar.set_position((0.8, .2, .03, .4))
plt.title(F1+"/"+F2+" All Diff"); 
plt.savefig("Plots/"+F1+"_"+F2+"_Diff.png",dpi=100,bbox_inches = 'tight')
plt.show()



####

for NN in Names:
    sns.kdeplot(data=aaa,x=NN,hue='Type',fill=True)
    plt.title(F1+"/"+F2+" "+NN)
    plt.savefig("Plots/"+F1+"_"+F2+"_"+NN+"_Hist.png",dpi=100,bbox_inches = 'tight')
    plt.show()    

    

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
 #'H2A',
 'pHistone H3 [S28]']

dfb=df
sns.set_style({'legend.frameon':True})
df=np.mean(aaa[aaa['Type']=="Mutant"][Names]).sort_values()
s1=np.full((len(df),4),0,dtype=np.float64)
sz1=np.full(len(df),0,dtype=np.float64)
df2=np.mean(aaa[aaa['Type']=="Wild"][Names]).sort_values()
s2=np.full((len(df2),4),0,dtype=np.float64)
sz2=np.full(len(df2),0,dtype=np.float64)

Srt=df-df2
Srt=Srt.sort_values(ascending=False)


for i in range(0,len(df)):
    print(str(i)+" "+df.index[i])
    q=np.std(aaa[aaa['Type']=="Mutant"][df.index[i]])*1/1.25
    print(q)
    s1[i]=[float(q),0,0,q]
    sz1[i]=q
    print(str(i)+" "+df2.index[i])
    q=np.std(aaa[aaa['Type']=="Wild"][df2.index[i]])*1./1.25
    print(q)
    s2[i]=[0,0,float(q),q]
    sz2[i]=q
    
    
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.hlines(y=df.index, xmin=-5, xmax=5, color='gray', alpha=0.7, 
          linewidth=1, linestyles='dashdot')

#ax.scatter(y=df.index, x=df, s=900*np.power(sz1,2), c='red', alpha=0.7,
#           label="Mutant",)
#ax.scatter(y=df2.index, x=df2, s=900*np.power(sz2,2), c='blue', alpha=0.7,
#           label="Wild")

ax.scatter(y=Srt.index, x=df.loc[Srt.index], s=900*np.power(sz1,2), c='red', alpha=0.7,
           label="Mutant",)
ax.scatter(y=Srt.index, x=df2.loc[Srt.index], s=900*np.power(sz2,2), c='blue', alpha=0.7,
           label="Wild")


ax.vlines(x=0, ymin=0, ymax=len(df)-1, color='black', alpha=0.7, linewidth=2, linestyles='dotted')
plt.legend(fontsize=20,
           facecolor='White', framealpha=1,frameon=True)

ax.set_title('Mean of Distribution - '+F1+'/'+F2, fontdict={'size':22})
ax.set_xlim(-1.5, 1.5)
plt.savefig("Plots/"+F1+"_"+F2+"_Means.png",dpi=100)
plt.show()

df=dfb


### PairWise Plots



title=F1+' Open Chromatin'
plt.figure(figsize=(6, 5))    
sns.pairplot(df[Open_Chromatin],corner=True,plot_kws={"s": 3},diag_kind="kde")
plt.title(title)
plt.savefig("Plots/"+F1+"_PW_Open.png",dpi=100)
plt.show()

title=F1+' Closed Chromatin'
plt.figure(figsize=(6, 5))    
sns.pairplot(df[Closed_Chromatin],corner=True,plot_kws={"s": 3},diag_kind="kde")
plt.title(title)
plt.savefig("Plots/"+F1+"_PW_Closed.png",dpi=100)
plt.show()

CompVars=[
    'H4K16ac',
    'H3K4me1',
    'H3K4me3',
    'H2AK119ub',
    'H3K27me3',
    'Type']

size=min(len(dfmut),len(df))
EqSize=aaa[aaa['Type']=="Mutant"].iloc[0:size]
EqSize=EqSize.append(aaa[aaa['Type']=="Wild"].iloc[0:size])




title=F1+'/'+F2
plt.figure(figsize=(6, 5))    
sns.pairplot(EqSize[CompVars].sort_values("Type",ascending=False),
             corner=True,hue="Type",plot_kws={"s": 3},
        )
plt.title(title)
plt.savefig("Plots/"+F1+"_"+F2+"_PW.png",dpi=100)
plt.show()
