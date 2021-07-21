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

def statistic(dframe):
    return dframe.corr().loc[Var1,Var2]

df=pd.read_csv("control.csv")
dfmut=pd.read_csv("mutant.csv")



'''
dfbck=df
df=np.arcsinh(df)
ar=np.asarray(df)
'''
df=df[df['H4']>0]
df=df[df['H3.3']>0]
df=df[df['H3']>0]
dfmut=dfmut[dfmut['H4']>0]
dfmut=dfmut[dfmut['H3']>0]
dfmut=dfmut[dfmut['H3.3']>0]


#df=df[(df != 0).all(1)]

Var='H3K4me1'

df=np.arcsinh(df)
dfmut=np.arcsinh(dfmut)


#hist, bins = np.histogram(df[(df['H3K27ac']<4000)]['H3K27ac'], bins=150)
hist, bins = np.histogram(df[df[Var]>00][Var], bins=50)
#hist=hist/sum(hist)
err=np.sqrt(hist)

center = (bins[:-1] + bins[1:]) / 2

#histm, bins = np.histogram(dfmut[(dfmut['H3K27ac']<4000)]['H3K27ac'], bins=150)
histm, bins = np.histogram(dfmut[dfmut[Var]>00][Var], bins=50)
#histm=histm/sum(histm)
errm=np.sqrt(hist)
centerm = (bins[:-1] + bins[1:]) / 2
plt.figure(figsize=(6, 5))
plt.plot(center,hist,'r.')
plt.plot(centerm,histm,'b.')
plt.title(Var)

relchange=(np.mean(dfmut[Var])-np.mean(df[Var]))/np.sqrt((np.std(df[Var]))**2+(np.std(dfmut[Var]))**2)
print(relchange)

plt.figure(figsize=(6, 5))
plt.matshow(df.corr())
plt.colorbar()
plt.title('Control')

plt.figure(figsize=(6, 5))
plt.matshow(dfmut.corr())
plt.colorbar()
plt.title('Mutant')

plt.figure(figsize=(6, 5))
diff=(dfmut.drop(columns='H3K27M').corr(method='spearman')-df.corr(method='spearman'))
plt.matshow(diff)
plt.colorbar()
plt.title('Diff Correleations Spearman')

plt.figure(figsize=(6, 5))
diff=np.corrcoef(np.transpose(dfmut.drop(columns='H3K27M')))-np.corrcoef(np.transpose(df))
plt.matshow(diff)
plt.colorbar()
plt.title('Diff Correleations Pearson')


Var1='H4'
Var2='H3.3'

plt.figure(figsize=(6, 5))
plt.scatter(df[Var1],df[Var2],s=2,c='r')
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
plt.xlim([0,1])
plt.legend(['Control','Mutant'])
#ci_low, ci_high,theta_star = bp.bcanon_interval(dist, statistic, df,
#                                                B=2,return_samples=True)
#print('Correlation ',Var1,'/',Var2,' ',se)#ci_low,' ',ci_high)



Names=['H3','H3K36me3','H2B','H3K4me3','H3K36me2','H4K16Ac','H2AK119ub','H3K4me1','H3.3',
       'H3K64ac','H4','H3K27ac','H3K9ac', 'H2BK120ub','H3K27me3','H3K27M',	'H3K9me3']
#Linear Regression using H3, H3.3, H4
ddf=dfmut
m=np.mean(ddf)
s=np.std(ddf)
ddf=(ddf-m)/s

'''
Pred=Names[1]
X=ddf[['H3','H3.3']]
Y=ddf['H4']
reg = LinearRegression().fit(X, Y)
reg.score(X, Y)
res=ddf['H4']-reg.predict(ddf[['H3','H3.3']])
a=reg.predict(ddf[['H3','H3.3']])
new=(ddf.iloc[:,:].transpose() - a).transpose()
plt.hist(res,100)
'''



avConstMarkers=np.sum(ddf[['H3.3','H4','H3']]/3,axis=1)

ddf=ddf.subtract(avConstMarkers,axis=0)

ar=np.asarray(ddf)
tsne = TSNE(n_components=2, random_state=0)

mask = np.random.choice([False, True], len(ar), p=[0.8, 0.2])

X_2d = tsne.fit_transform(ar[mask,:])

TSNEVar='H3K27ac'
cc=ddf[TSNEVar][mask]
plt.figure(figsize=(6, 5))

plt.scatter(X_2d[:,0],X_2d[:,1],s=2,
            c=cc, cmap=plt.cm.Spectral)
plt.colorbar()
plt.clim(-5,5)
plt.title(TSNEVar)

TSNEVar='H3.3'
cc=ddf[TSNEVar][mask]
plt.figure(figsize=(6, 5))

plt.scatter(X_2d[:,0],X_2d[:,1],s=2,
            c=cc, cmap=plt.cm.Spectral)
plt.colorbar()
plt.clim(-5,5)
plt.title(TSNEVar)


'''

plt.figure(figsize=(6, 5))

plt.scatter(X_2d[:,0],X_2d[:,1],s=2,
            c=df['H3.3'][mask], cmap=plt.cm.Spectral)
plt.colorbar()
plt.title('H3.3')


plt.figure(figsize=(6, 5))
plt.scatter(X_2d[:,0],X_2d[:,1],s=2,
            c=df['H4'][mask], cmap=plt.cm.Spectral)
plt.colorbar()
plt.title('H4')
'''