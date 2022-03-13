# -*- coding: utf-8 -*-
"""
@author: Anderson
"""

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class nmc(BaseEstimator, ClassifierMixin):
    def __init__(self, k=1):
        self.k = k
# Função fit faz a média dos pontos de cada classe  
    def fit(self, x_train, y_train):
        qclasses=np.unique(y_train)
        cluster_mean=np.zeros((qclasses.size,x_train.shape[1]))
        count=0
        for i in qclasses:
            yindex=y_train==i
            xclasse=x_train[yindex]
            cluster_mean[count,:]=xclasse.mean(axis=0)
            count+=1
        
        self.qclasses=qclasses
        self.cluster_mean=cluster_mean
        self.X_train = x_train
        self.y_train = y_train
        return self
# Função predict determina a distancia de cada ponto do x_test ao centro do cluster(Média dos pontos) de cada classe,
# a menor distancia determina a que classe o ponto deve pertencer .   
    def predict(self, x_test):
        m=x_test.shape[0]
        n=self.qclasses.size
        distance=np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                a=x_test[i,:]
                b=self.cluster_mean[j,:]
                dist = np.sqrt(np.sum(np.square(a-b)))
                distance[i,j]=dist
        y_pred=distance.argmin(axis=1)
                
        return y_pred       
    
    #Funções necessarias para a utilização do Gridsearch e Pipeline
    def score(self, x_test, y_test):
        predictions = self.predict(x_test)
        return (predictions == y_test).sum() / len(y_test)
    
    def get_params(self, deep=True):
        
        return {"k": self.k}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self                