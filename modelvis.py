import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from tree import DecisionTreeClassifier

def plot_classifier_2d(clf, data, target):
    x_min, x_max = data.iloc[:,0].min(), data.iloc[:,0].max()
    y_min, y_max = data.iloc[:,1].min(), data.iloc[:,1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min)/100), np.arange(y_min, y_max, (y_max - y_min)/100))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,0]
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.magma, alpha = 0.5)
    plt.scatter(x = data.iloc[:,0], y = data.iloc[:,1], c = target, s = 50, alpha = 0.3)
    plt.colorbar(cs)

if __name__ == "__main__":
    print("welcome to model visualisation")

