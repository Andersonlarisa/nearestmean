# -*- coding: utf-8 -*-
"""

@author: Anderson
"""

from sklearn.datasets import make_blobs
import numpy as np
from nearestmean import nmc
import matplotlib.pyplot as plt
import streamlit as st

st.title('Exemplo Nearest Mean')

qtd_Amostras=st.sidebar.number_input('qtd Amostras',value=100,min_value=2)
qtd_Classes=st.sidebar.number_input('qtd Classes',value=3,min_value=2)
desvio=st.sidebar.number_input('Desvio Padrão',value=0.4,min_value=0.0)
random_state=st.sidebar.number_input('Random State',value=0,min_value=0)


x, y = make_blobs(n_samples=qtd_Amostras, centers=qtd_Classes, n_features=2,cluster_std=desvio,
                  random_state=random_state)


clf=nmc()
clf.fit(x,y)

x1=st.sidebar.number_input('Amostra eixo x',value=1)
y1=st.sidebar.number_input('Amostra eixo y',value=1)
x_test=np.array([x1,y1],ndmin=2)

y_pred=clf.predict(x_test)

fig, ax = plt.subplots()
ax.scatter(x[:,0], x[:,1],c=y)
ax.scatter(clf.cluster_mean[:,0],clf.cluster_mean[:,1],c='#000000')
ax.scatter(x_test[:,0],x_test[:,1],c=y_pred,marker='x',vmin=y.min(),vmax=y.max())

#plt.show()
st.pyplot(fig)
