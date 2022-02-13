from Network    import Network;
from Test_Train import Test, Train;
from Points     import Generate_Coords, Generate_Boundary;
from Plotter import Update_Axes, Generate_Plot_Gridpoints, Setup_Axes; # This is really bad code.

import torch;
import numpy;
import random;
import matplotlib.pyplot as plt;



# Hyper-parameters.
num_hidden_layers       : int   = 5;
units_per_layer         : int   = 20;

lr                      : float = 0.0025;
num_epochs              : int   = 2500;

Num_Train_Coll_Coords   : int   = 2000;
Num_Train_BC_Coords     : int   = 500;      # This is per side of the boundary.

Num_Test_Coll_Coords    : int   = 1000;
Num_Test_BC_Coords      : int   = 250;      # This is per side of the boundary.


# Specify problem domain. We will enforce the Poission equation for
# (x, y) \in [x_low, x_high] x [y_low, y_high]
x_low   : float = -1.0;
x_high  : float = 1.0;
y_low   : float = -1.0;
y_high  : float = 1.0;



# Class for driving term in Poission equation.
class Driving_Term(torch.nn.Module):
    def __init__(self):
        # Call the Module constructor.
        super().__init__();

    def forward(self, Coords : torch.Tensor) -> torch.Tensor:
        # Coords should be a N by 2 tensor whose ith row holds the x, y
        # components of the ith coordinate.

        X : torch.Tensor = Coords[:, 0];
        Y : torch.Tensor = Coords[:, 1];

        # Evaluate the driving term at the coordinates.
        return torch.mul(torch.sin(numpy.pi * X), torch.sin(numpy.pi * Y));



def main():
    ############################################################################
    # Setup

    # First, set up U.
    U = Network(    num_hidden_layers = num_hidden_layers,
                    units_per_layer   = units_per_layer);

    # Next, set up the optimizer. The params attribute is the set of all
    # trainble parameters in U (torch has a special Tensor sub-class called a
    # "paramater" which is basically just a Tensor that requires grad by
    # defualt. If M is a module object, then M.parameters). You can easily
    # obtain an itterator over a module's paramater attributes via the
    # "paramaters" method.
    Optim = torch.optim.Adam(params = U.parameters(), lr = lr);

    # Generate the Collocation coordinates.
    Train_Coll_Coords : torch.Tensor = Generate_Coords(
                                                 x_l = x_low,
                                                 x_h = x_high,
                                                 y_l = y_low,
                                                 y_h = y_high,
                                                 num_coords = Num_Train_Coll_Coords);
    Test_Coll_Coords : torch.Tensor = Generate_Coords(
                                                 x_l = x_low,
                                                 x_h = x_high,
                                                 y_l = y_low,
                                                 y_h = y_high,
                                                 num_coords = Num_Test_Coll_Coords);


    # Generate the boundary coordinates, targets.
    (Train_BC_Coords, Train_BC_Targets) = Generate_Boundary(
                                                 x_l = x_low,
                                                 x_h = x_high,
                                                 y_l = y_low,
                                                 y_h = y_high,
                                                 num_coords = Num_Train_BC_Coords);

    (Test_BC_Coords, Test_BC_Targets) = Generate_Boundary(
                                                 x_l = x_low,
                                                 x_h = x_high,
                                                 y_l = y_low,
                                                 y_h = y_high,
                                                 num_coords = Num_Test_BC_Coords);


    # Finally, up the driving term.
    f = Driving_Term();



    ############################################################################
    # Epochs!

    # Run the epochs!!!
    for t in range(num_epochs):
        print("Epoch %4u: " % (t + 1), end = '');

        # First, we train!
        Train(  U           = U,
                f           = f,
                Coll_Coords = Train_Coll_Coords,
                BC_Coords   = Train_BC_Coords,
                BC_Targets  = Train_BC_Targets,
                Optim       = Optim);

        # Now, test.
        Test(   U           = U,
                f           = f,
                Coll_Coords = Test_Coll_Coords,
                BC_Coords   = Test_BC_Coords,
                BC_Targets  = Test_BC_Targets);



    ############################################################################
    # Plot

    # Set up figure and Axes.
    fig, Axes = Setup_Axes();

    # Set up plotting gridpoints.
    Plotting_Coords = Generate_Plot_Gridpoints( x_l = x_low,
                                                x_h = x_high,
                                                y_l = y_low,
                                                y_h = y_high,
                                                n   = 50);

    # Plot final results.
    Update_Axes(    fig     = fig,
                    Axes    = Axes,
                    U       = U,
                    f       = f,
                    Coords  = Plotting_Coords,
                    n       = 50);
    plt.show();


if __name__ == "__main__":
    main();
