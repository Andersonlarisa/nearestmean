# -*- coding: utf-8 -*-
"""

@author: Anderson
"""

from sklearn.datasets import make_blobs
import numpy as np
from nearestmean import nmc
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split

st.title('Exemplo Nearest Mean')

st.sidebar.write('Parâmetros do Dataset')
qtd_Amostras=st.sidebar.number_input('Qtd Amostras',value=100,min_value=2)
qtd_Classes=st.sidebar.number_input('Qtd Classes',value=3,min_value=2)
desvio=st.sidebar.number_input('Desvio Padrão',value=0.4,min_value=0.0)
random_state=st.sidebar.number_input('Random State',value=0,min_value=0)


x, y = make_blobs(n_samples=qtd_Amostras, centers=qtd_Classes, n_features=2,cluster_std=desvio,
                  random_state=random_state)

x, x_test, y, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

clf=nmc()
clf.fit(x,y)

nc=NearestCentroid()
nc.fit(x,y)



#st.sidebar.write('Amostra para classificação')
#x1=st.sidebar.number_input('Amostra eixo x',value=1)
#y1=st.sidebar.number_input('Amostra eixo y',value=2)
#x_test=np.array([x1,y1],ndmin=2)

y_proba_nmc=clf.predict_proba(x_test)
y_pred_nc=nc.predict(x_test)
y_pred_nmc=clf.predict(x_test)

print(y_pred_nc==y_pred_nmc)

#fig, ax = plt.subplots()
#ax.scatter(x[:,0], x[:,1],c=y)
#ax.scatter(clf.cluster_mean[:,0],clf.cluster_mean[:,1],c='#000000')
#ax.scatter(x_test[:,0],x_test[:,1],c=y_pred_nc,marker='x',vmin=y.min(),vmax=y.max())
#plt.show()
#st.pyplot(fig)
