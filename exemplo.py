import sklearn.utils.estimator_checks
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from nearestmeanclassifier import KNN
import numpy as np
from sklearn.model_selection import GridSearchCV

x, y = make_blobs(n_samples=100, centers=4, n_features=2,cluster_std=0.4,
                  random_state=0)
print(x.shape)
print(y.shape)
# plot
fig, ax = plt.subplots()

ax.scatter(x[:,0], x[:,1],c=y)

#plt.show()
xtest=np.array([1,1])
xtest=xtest.reshape(1,2)

# knn=KNN(3)

# knn.fit(x,y)
#pred=knn.predict(xtest)

params = {
    'k': [2, 3,4]
}

gs = GridSearchCV(KNN(), param_grid=params, cv=4)
gs.fit(x,y)
#sklearn.utils.estimator_checks.check_estimator(knn)
#print(pred)



