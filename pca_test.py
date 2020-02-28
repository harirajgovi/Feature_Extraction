# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:07:46 2019

@author: hari4
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv("Wine.csv")
x_mtx = dataset.iloc[:, :-1].values
y_vctr = dataset.iloc[:, -1].values

#splitting to train test data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_mtx, y_vctr, test_size=0.20, random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_

# Running Logistic Regression Classsifier models
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

y_prdc = classifier.predict(x_test)

#importing confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_prdc)

#Visualizing the results
from matplotlib.colors import ListedColormap
x_learn, y_learn = x_train, y_train
x1, x2 = np.meshgrid(np.arange(x_learn[:, 0].min() - 1,x_learn[:, 0].max() + 1, 0.01),
                     np.arange(x_learn[:, 1].min() - 1,x_learn[:, 1].max() + 1, 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(("red", "green", "blue")))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_learn)):
    plt.scatter(x_learn[y_learn == j, 0], x_learn[y_learn == j, 1],
                color=ListedColormap(("yellow", "pink", "cyan"))(i), label=j)
    
plt.title("Logistic Regression (Training set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()

#Visualizing the results
from matplotlib.colors import ListedColormap
x_learn, y_learn = x_test, y_test
x1, x2 = np.meshgrid(np.arange(x_learn[:, 0].min() - 1,x_learn[:, 0].max() + 1, 0.01),
                     np.arange(x_learn[:, 1].min() - 1,x_learn[:, 1].max() + 1, 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(("red", "green", "blue")))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_learn)):
    plt.scatter(x_learn[y_learn == j, 0], x_learn[y_learn == j, 1],
                color=ListedColormap(("yellow", "pink", "cyan"))(i), label=j)
    
plt.title("Logistic Regression (Test set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()




