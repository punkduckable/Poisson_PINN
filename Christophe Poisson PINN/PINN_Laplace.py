import torch
import torch.autograd
import torch.nn
#import caffeine

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from MiscFunctions import *

import warnings

warnings.filterwarnings('ignore')

#caffeine.on(display = True) # Prevents computer to turn into sleep mode if working on MacOS

torch.manual_seed(0)

class PhysicsInformedNet(torch.nn.Module):

    # Torch-defined neural network

    def __init__(self, n_units):
        super(PhysicsInformedNet, self).__init__()

        linear1 = torch.nn.Linear(2, n_units)
        linear2 = torch.nn.Linear(n_units, n_units)
        linear3 = torch.nn.Linear(n_units, n_units)
        linear4 = torch.nn.Linear(n_units, 1)

        torch.nn.init.xavier_uniform_(linear1.weight)
        torch.nn.init.xavier_uniform_(linear2.weight)
        torch.nn.init.xavier_uniform_(linear3.weight)
        torch.nn.init.xavier_uniform_(linear4.weight)

        self.fc1 = linear1
        self.fc2 = linear2
        self.fc3 = linear3
        self.fc4 = linear4

    def forward(self, x):

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)

        return x


def rhs(x):

    # Right-Hand-Side of the Poisson equation

    return 2 * np.pi ** 2 * torch.sin(1 * np.pi * x[0, 0]) * torch.sin(1 * np.pi * x[0, 1])


def trueSolution(x, y):

    return np.sin(1 * np.pi * x) * np.sin(1 * np.pi * y)


def poissonLoss(net, X_in, X_bound, g):

    # Physics-informed loss function

    # Equation:
    # u_xx + u_yy = -q on [0, 1]x[0, 1]
    # u = g on Boundaries

    loss_in = 0
    loss_bound = 0
    n_in = X_in.shape[0]
    n_bound = X_bound.shape[0]

    # Collocation points inside the domain

    for i in range(n_in):

        x = X_in[i:i + 1, :]
        x_detached = x.detach()
        x_detached.requires_grad = True
        u = net.forward(x_detached)

        grad_u = torch.autograd.grad(u, x_detached, create_graph = True, retain_graph = True)[0]
        u_x = grad_u[0, 0]
        u_y = grad_u[0, 1]

        u_xx = torch.autograd.grad(u_x, x_detached, create_graph = True, retain_graph = True)[0][0, 0]
        u_yy = torch.autograd.grad(u_y, x_detached, create_graph = True, retain_graph = True)[0][0, 1]

        q = rhs(x)
        temp = (u_xx + u_yy + q) ** 2
        loss_in = torch.add(loss_in, temp)

    # Collocation points on the boundary (enforcing Dirichlet conditions)

    for i in range(n_bound):

        x = X_bound[i:i + 1, :]
        x_detached = x.detach()
        x_detached.requires_grad = True
        u = net.forward(x_detached)

        temp = (u - g) ** 2
        loss_bound = torch.add(loss_bound, temp)

    loss_in = (loss_in / n_in) ** 0.5
    loss_bound = (loss_bound / n_bound) ** 0.5

    loss = torch.add(loss_in, loss_bound)

    return loss


def residual(net, x, y):

    X_in = torch.tensor(np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))).type(torch.FloatTensor)
    X_in.requires_grad = True
    R = []

    for i in range(X_in.shape[0]):

        x = X_in[i:i + 1, :]
        x_detached = x.detach()
        x_detached.requires_grad = True
        u = net.forward(x_detached)

        grad_u = torch.autograd.grad(u, x_detached, create_graph = True, retain_graph = True)[0]
        u_x = grad_u[0, 0]
        u_y = grad_u[0, 1]

        u_xx = torch.autograd.grad(u_x, x_detached, create_graph = True, retain_graph = True)[0][0, 0]
        u_yy = torch.autograd.grad(u_y, x_detached, create_graph = True, retain_graph = True)[0][0, 1]

        q = rhs(x)
        r = (u_xx + u_yy + q)
        r = np.abs(r.detach().numpy())

        R.append(r)

    R = np.array(R)

    return R


def generateData(n_in, n_bound):

    # Generating random collocation points

    X_in = torch.rand(n_in, 2, requires_grad = True)

    n_bound = int(n_bound / 4) - 1

    X_bound1 = torch.cat((torch.rand(n_bound, 1, requires_grad = True), 0 * torch.ones(n_bound, 1)), 1)
    X_bound2 = torch.cat((torch.rand(n_bound, 1, requires_grad = True), 1 * torch.ones(n_bound, 1)), 1)
    X_bound3 = torch.cat((0 * torch.ones(n_bound, 1), torch.rand(n_bound, 1, requires_grad = True)), 1)
    X_bound4 = torch.cat((1 * torch.ones(n_bound, 1), torch.rand(n_bound, 1, requires_grad = True)), 1)
    X_boundCorners = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], requires_grad = True)

    X_bound = torch.cat((X_bound1, X_bound2, X_bound3, X_bound4, X_boundCorners), 0)

    return X_in, X_bound


def plot(fig, net, x_star, y_star, n, loss_list, i):

    # Plots the current neural net prediction, the PDE residual, the true solution and the loss

    Unet = evalNet(net, x_star, y_star)
    R = residual(net, x_star, y_star)
    Utrue = trueSolution(x_star, y_star)

    if fig is None:

        fig = plt.figure(figsize = (10, 8))
        plt.subplot(221)
        plt.contourf(x_star.reshape(n, n), y_star.reshape(n, n), Unet.reshape(n, n), 50, cmap = plt.cm.jet)
        plt.axis('equal')
        plt.title('NN(x, y)')
        plt.colorbar()

        plt.subplot(222)
        plt.contourf(x_star.reshape(n, n), y_star.reshape(n, n), R.reshape(n, n), 50, cmap = plt.cm.jet)
        plt.axis('equal')
        plt.title('Residual')
        plt.colorbar()

        plt.subplot(223)
        plt.contourf(x_star.reshape(n, n), y_star.reshape(n, n), Utrue.reshape(n, n), 50, cmap = plt.cm.jet)
        plt.axis('equal')
        plt.title('True Solution')
        plt.colorbar()

        plt.subplot(224)
        plt.plot(np.arange(0, i + 1, 1), loss_list)
        plt.xlim([0, n_iter])
        plt.title('Loss')

        plt.pause(0.0001)

        return fig

    else:

        plt.clf()
        plt.subplot(221)
        plt.contourf(x_star.reshape(n, n), y_star.reshape(n, n), Unet.reshape(n, n), 50, cmap=plt.cm.jet)
        plt.axis('equal')
        plt.title('NN(x, y)')
        plt.colorbar()

        plt.subplot(222)
        plt.contourf(x_star.reshape(n, n), y_star.reshape(n, n), R.reshape(n, n), 50, cmap=plt.cm.jet)
        plt.axis('equal')
        plt.title('Residual')
        plt.colorbar()

        plt.subplot(223)
        plt.contourf(x_star.reshape(n, n), y_star.reshape(n, n), Utrue.reshape(n, n), 50, cmap=plt.cm.jet)
        plt.axis('equal')
        plt.title('True Solution')
        plt.colorbar()

        plt.subplot(224)
        plt.plot(np.arange(0, i + 1, 1), loss_list)
        plt.xlim([0, n_iter])
        plt.title('Loss')

        plt.show(block = False)
        fig.canvas.draw()
        plt.pause(0.0001)

        return fig



n_units = 50
net = PhysicsInformedNet(n_units) # Defines the PINN

n_in = 500
n_bound = 400
g = 0

X_in, X_bound = generateData(n_in, n_bound) # Creates the collocation data

net.train() # In Torch, neural networks should be set on train() mode before gradient descent

n_iter = 10
optim = torch.optim.Adam(net.parameters(), lr = 0.0001) # Gradient descent optimizer

fig = None
n = 20
x_star = np.linspace(0, 1, n)
y_star = np.linspace(0, 1, n)
x_star, y_star = np.meshgrid(x_star, y_star)
loss_list = []

for i in range(n_iter):

    optim.zero_grad() # Reset the gradient descent gradients to zero
    loss = poissonLoss(net, X_in, X_bound, g) # Computes physics informed loss
    loss.backward(retain_graph = False) # Computes loss gradient w.r.t the PINN weights
    optim.step() # Applies the gradients to the current weights

    print('Iter: %d, Loss: %2.6f' %(i, loss.item()))
    loss_list.append(loss.item())

    if i % 50 == 0:
        fig = plot(fig, net, x_star, y_star, n, loss_list, i)

net.eval()


n = 50
x_star = np.linspace(0, 1, n)
y_star = np.linspace(0, 1, n)
x_star, y_star = np.meshgrid(x_star, y_star)
Unet = evalNet(net, x_star, y_star)
R = residual(net, x_star, y_star)

plotGrid(x_star, y_star, Unet, n)
plotGrid(x_star, y_star, R, n)

debug_flag = 1
