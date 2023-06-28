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
        classes=np.unique(y_train)
        cluster_mean=np.zeros((classes.size,x_train.shape[1]))
        count=0
        for i in classes:  # Trocar para enumerate
            yindex=y_train==i
            xclasse=x_train[yindex]
            cluster_mean[count,:]=xclasse.mean(axis=0)
            count+=1
        
        self.classes=classes   # array classes
        self.cluster_mean=cluster_mean  # Posição do centro de cada classe
        self.X_train = x_train
        self.y_train = y_train
        return self
# Função predict determina a distancia de cada ponto do x_test ao centro do cluster(Média dos pontos) de cada classe,
# a menor distancia determina a que classe o ponto deve pertencer .   
    def predict(self, x_test):
        m=x_test.shape[0]
        n=self.classes.size
        distance=np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                a=x_test[i,:]
                b=self.cluster_mean[j,:]
                dist = np.sqrt(np.sum(np.square(a-b)))
                distance[i,j]=dist
        y_pred=distance.argmin(axis=1)
                
        return y_pred
    
    def predict_proba(self, x_test):
        n=self.classes.size # Para determinar a probabilidade o k deve ser igual a quantidade de classes
        m=x_test.shape[0]
        distance=np.zeros((m,n))
        probabilidades=[]

        for i in range(m):
            for j in range(n):
                a=x_test[i,:]
                b=self.cluster_mean[j,:]
                dist = np.sqrt(np.sum(np.square(a-b)))
                distance[i,j]=dist
        
        with np.errstate(divide='ignore'):
            peso = 1. / distance
        inf_mask = np.isinf(peso)
        inf_row = np.any(inf_mask, axis=1)
        peso[inf_row] = inf_mask[inf_row]
        
        normalizador=peso.sum(axis=1)[:, np.newaxis]
        normalizador[normalizador == 0.0] = 1.0
        peso /= normalizador
        #probabilidades.append(peso)
        probabilidades=peso
        
        return probabilidades
    
    #Funções necessarias para a utilização do Gridsearch e Pipeline.
    def score(self, x_test, y_test):
        predictions = self.predict(x_test)
        return (predictions == y_test).sum() / len(y_test)
    
    def get_params(self, deep=True):
        
        return {"k": self.k}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self                
