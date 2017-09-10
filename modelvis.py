import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_classifier(clf, X, y):
    x1_min, x1_max = X.iloc[:,0].min(), X.iloc[:,0].max()
    x2_min, x2_max = X.iloc[:,1].min(), X.iloc[:,1].max()
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, (x1_max - x1_min)/100),
            np.arange(x2_min, x2_max, (x2_max - x2_min)/100))

    Z = clf.predict_proba(np.c_[xx1.ravel(), xx2.ravel()])[:,0]
    Z = Z.reshape(xx1.shape)
    cs = plt.contourf(xx1, xx2, Z, cmap=plt.cm.magma, alpha = 0.5)
    plt.scatter(x = X.iloc[:,0], y = X.iloc[:,1], c = y, s = 50, alpha = 0.3)
    plt.colorbar(cs)

if __name__ == "__main__":
    main()
