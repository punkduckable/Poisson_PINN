from Network import Network;
from Losses import PDE_Residual;

import matplotlib.pyplot as plt;
import torch;
import numpy;
from typing import Tuple;



# Generate plotting gridpoints.
def Generate_Plot_Gridpoints(   x_l : float,
                                x_h : float,
                                y_l : float,
                                y_h : float,
                                n   : int) -> torch.Tensor:
    """ This function generates a uniformly spaced grid of points in [0,1]x[0,1]
    for plotting purposes. Sure, you could do this with meshgrid, but I prefer
    to write my own code, especially if it won't impact performance by much
    (the code should only call this function once). I like explicit code.

    ----------------------------------------------------------------------------
    Arguments:

    n : the number of grid points in each dimension. This function will generate
    an n by n grid of points in the unit (2) square.

    ----------------------------------------------------------------------------
    Returns:

    an n^2 x 2 tensor of points. The ith row of this tensor holds the x,y
    coordinates of the ith point. The zero element has coordinates (0,0),
    the 1 element has coordinates (0, 1/(n - 1))... the n element has coordinates
    (1/(n - 1), 0)... the n*n - 1 element has coordinates (1, 1). """

    Coords = torch.empty((n*n, 2), dtype = torch.float);

    # Using the rule defined above, we can see that the i*n + j element of
    # Coords should have coordinates j/(n - 1), i/(n - 1). (think about it)
    for i in range(n):
        for j in range(n):
            Coords[n*i + j, 0] = i/(n - 1);
            Coords[n*i + j, 1] = j/(n - 1);

    # Rescale the points.
    Coords[:, 0] = (x_h - x_l)*Coords[:, 0] + x_l;
    Coords[:, 1] = (y_h - y_l)*Coords[:, 1] + y_l;

    return Coords;



# Set up Axes objects for plotting
def Setup_Axes() -> Tuple[plt.figure, numpy.array]:
    """ This function sets up the figure, axes objects for plotting. There
    are a lot of settings to tweak, so I thought the code would be cleaner
    if those details were outsourced to this function.

    ----------------------------------------------------------------------------
    Arguments:
    None!

    ----------------------------------------------------------------------------
    Returns:
    A tuple. The first element contains the figure object, the second contains
    a numpy array of axes objects (to be passed to Update_Axes). """

    # Set up the figure object.
    fig = plt.figure(figsize = (8, 4));

    # Approx solution subplot.
    Axes1 = fig.add_subplot(1, 2, 1);
    Axes1.set_title("Neural Network Approximation");
    Axes1.set_xlabel("x (m)");
    Axes1.set_ylabel("y (m)");
    Axes1.set_aspect('equal', adjustable = 'datalim');
    Axes1.set_box_aspect(1.);

    # Residual subplot.
    Axes2 = fig.add_subplot(1, 2, 2);
    Axes2.set_title("PDE Residual");
    Axes2.set_xlabel("x (m)");
    Axes2.set_ylabel("y (m)");
    Axes2.set_aspect('equal', adjustable = 'datalim');
    Axes2.set_box_aspect(1.);

    # Package axes objects into an array.
    Axes = numpy.array([Axes1, Axes2]);

    return (fig, Axes);



# The plotting function!
def Update_Axes(    fig     : plt.figure,
                    Axes    : numpy.array,
                    U       : Network,
                    f       : torch.nn.Module,
                    Coords  : torch.Tensor,
                    n       : int) -> None:
    """ This function plots the approximate solution and residual at the
    specified points.

    ----------------------------------------------------------------------------
    Arguments:

    fig : The figure object to which the Axes belong. We need this to set up
    the color bars.

    Axes : The array of Axes object that we will plot on. Note that this
    function will overwrite these axes.

    U : The neural network that gives the approximate solution to Poisson's
    equation.

    Coords : The set of points we want to evaluate the approximate and true
    solutions, as well as the PDE Residual. This should be an (n*n)x2 tensor,
    whose ith row holds the x,y coordinates of the ith point we want to plot.
    Each element of Coords should be an element of [0,1]x[0,1].

    n : the number of gridpoints along each axis. Coords should be an n*n x 2
    tensor.

    ----------------------------------------------------------------------------
    Returns:

    Nothing! """

    # First, evaluate the network's approximate solution, the true solution, and
    # the PDE residual at the specified Coords. We need to reshape these into
    # nxn grids, because that's what matplotlib's contour function wants. It's
    # annoying, but it is what it is.
    U_Grid        = U(Coords).detach().numpy().reshape(n, n);
    Residual_Grid = PDE_Residual(   U       = U,
                                    f       = f,
                                    Coords  = Coords).detach().numpy().reshape(n, n);

    # Extract the x and y coordinates of points, as np arrays. We also need to
    # reshape these as nxn grids (same reason as above.
    X_Grid = Coords[:, 0].detach().numpy().reshape(n,n);
    Y_Grid = Coords[:, 1].detach().numpy().reshape(n,n);

    # Plot the approximate solution + colorbar.
    ColorMap0 = Axes[0].contourf(X_Grid, Y_Grid, U_Grid.reshape(n,n), levels = 300, cmap = plt.cm.jet);
    fig.colorbar(ColorMap0, ax = Axes[0], fraction=0.046, pad=0.04, orientation='vertical');

    # Plot the residual + colorbar
    ColorMap1 = Axes[1].contourf(X_Grid, Y_Grid, Residual_Grid.reshape(n,n), levels = 300, cmap = plt.cm.jet);
    fig.colorbar(ColorMap1, ax = Axes[1], fraction=0.046, pad=0.04, orientation='vertical');

    # Set tight layout (to prevent overlapping... I have no idea why this isn't
    # a default setting. Matplotlib, you are special kind of awful).
    fig.tight_layout();
