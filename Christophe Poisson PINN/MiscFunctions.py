import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import Delaunay


def evalNet(net, x, y):

    X_star = torch.tensor(np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))).type(torch.FloatTensor)
    U = []

    for i in range(X_star.shape[0]):

        u = net.forward(X_star[i:i + 1, :])[0, 0].detach().numpy()
        U.append(u)

    U = np.array(U)

    return U


def plotGrid(x, y, U, n):

    if U.ndim != 2:
        U = U.reshape(n, n)

    plt.figure()
    plt.contourf(x.reshape(n, n), y.reshape(n, n), U, 100, cmap = plt.cm.jet)
    plt.axis('equal')
    plt.colorbar()


def plotRandom(X, U):

    x_train, y_train = X[:, 0], X[:, 1]
    x_train = x_train.detach().numpy()
    y_train = y_train.detach().numpy()
    triang = Delaunay(np.hstack((x_train.reshape(-1, 1), y_train.reshape(-1, 1)))).vertices
    triang = mtri.Triangulation(x_train, y_train, triang)
    x_train, y_train, np.meshgrid(x_train, y_train)

    plt.figure()
    plt.tricontourf(triang, U, 100, cmap = plt.cm.jet)
    plt.axis('equal')
    plt.colorbar()
