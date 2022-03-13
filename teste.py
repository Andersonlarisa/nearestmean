# -*- coding: utf-8 -*-
"""

@author: Anderson
"""

from sklearn.datasets import make_blobs
import numpy as np
from nearestmean import nmc
import matplotlib.pyplot as plt

x, y = make_blobs(n_samples=100, centers=5, n_features=2,cluster_std=0.4,
                  random_state=0)


clf=nmc()
clf.fit(x,y)

x_test=np.array([[1,1],[-1,6]])

y_pred=clf.predict(x_test)

fig, ax = plt.subplots()
ax.scatter(x[:,0], x[:,1],c=y)
ax.scatter(clf.cluster_mean[:,0],clf.cluster_mean[:,1],c='#000000')
ax.scatter(x_test[:,0],x_test[:,1],c=y_pred,marker='x',vmin=y.min(),vmax=y.max())

plt.show()

