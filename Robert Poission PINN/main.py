import torch;
import numpy as np;
from typing import Tuple;
import matplotlib.pyplot as plt;

from Poisson_PINN import Neural_Network, f, Collocation_Loss, Boundary_Loss;
from Plotter import Update_Axes, Generate_Plot_Gridpoints, Setup_Axes;
from Setup_File_Reader import Setup_File_Reader, Setup_Data_Container;



# Training Loop
def Training_Loop(  u_NN                : Neural_Network,
                    Collocation_Points  : torch.Tensor,
                    Boundary_Points     : torch.Tensor,
                    Optimizer           : torch.optim.Optimizer) -> None:
    """ This loop runs one epoch of training for the neural network. In
    particular, we enforce the PDE at the specified Collocation_Points, and the
    boundary conditions at the Boundary_Points.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : The neural network that is approximating the solution to the PDE.

    Collocation_Points : the collocation points at which we want to enforce the
    PDE. These should be on the interior of the domain. Futher, these should be
    DISTINCT from the points we test the network at. This should be an Nx2
    tensor of floats, where N is the number of collocation points. The ith
    row of this tensor should be the coordinates of the ith collocation point.

    Boundary_Points : The points on the boundary at which we want to enforce the
    boundary conditions. This should be a Bx2 tensor of floats, where B is
    the number of boundary points. The ith row of this tensor should hold the
    coordinates of the ith boundary point.

    optimizer : the optimizer we use to train u_NN.

    ----------------------------------------------------------------------------
    returns:
    Nothing! """

    num_Collocation_Points : int = Collocation_Points.shape[0];
    num_Boundary_Points    : int = Boundary_Points.shape[0];

    # Zero out the gradients in the neural network.
    Optimizer.zero_grad();

    # Evaluate the Loss (Note, we enforce a BC of 0)
    Loss = (Collocation_Loss(u_NN, Collocation_Points) +
            Boundary_Loss(u_NN, Boundary_Points, 0));

    # Back-propigate to compute gradients of Loss with respect to network
    # weights.
    Loss.backward();

    # update network weights.
    Optimizer.step();



# Testing Loop
def Testing_Loop(   u_NN                : Neural_Network,
                    Collocation_Points  : torch.Tensor,
                    Boundary_Points     : torch.Tensor) -> Tuple[float, float]:
    """ This loop tests the neural network at the specified Boundary and
    Collocation points. You CAN NOT run this function with no_grad set True.
    Why? Because we need to evaluate derivatives of the solution with respect
    to the inputs! This is a PINN, afterall. This means that we need torch to
    build a computa1tional graph.

    Should we worry about the computational graph getting stuck in limbo/not
    being cleaned up (we don't call backward in here, which is usually what
    fees it)? No! Pytorch will construct a graph for the losses. However,
    once this function retuns, the losses are destroyed (they're automatic
    variables), along with their graphs.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : The neural network that approximates the PDE solution.

    Collocation_Points : the collocation points at which we want to enforce the
    PDE. These should be on the interior of the domain. Futher, these should be
    DISTINCT from the collocation points that we use in the training loop.
    This should be an Nx2 tensor of floats, where N is the number of collocation
    points. The ith row of this tensor should be the coordinates of the ith
    collocation point.

    Boundary_Points : The points on the boundary at which we want to enforce the
    boundary conditions. These should be DISTINCT from the boundary points we
    use in the training loop. This should be a Bx2 tensor of floats, where B is
    the number of boundary points. The ith row of this tensor should hold the
    coordinates of the ith boundary point.

    ----------------------------------------------------------------------------
    Returns:
    a tuple of floats. The first element holds the collocation loss, while
    the second holds the boundary loss. """

    # Get the losses at the passed collocation points (Note we enforce a 0 BC)
    Coloc_Loss : float = Collocation_Loss(u_NN, Collocation_Points).item();
    Bound_Loss : float = Boundary_Loss(u_NN, Boundary_Points, 0).item();

    # Return the losses.
    return (Coloc_Loss, Bound_Loss);

    # Should we worry about the computational graph that we build in this
    # function? No. Here's why:
    # Cmputing the losses requires propigating the inputs through the network,
    # thereby building up a computational graph (we need to keep the graph
    # building enabled b/c we have to evaluate derivatives to compute the
    # collocation loss). Normally, these graphs are freed when we call backward.
    # That's what happens in the training loop. Here, we don't use backward.
    # The graphs will be freed, however. This function builds up graphs for
    # Coloc_Loss and Bound_Loss. When this function returns, however, both
    # variables are freed (along with their graphs!). These graphs do not get
    # passed Coloc_Loss or Bound_Loss, since both are floats (not Tensors).



# Generate collocation and boundary points
def generate_points(num_Collocation_Points : int, num_Boundary_Points : int) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Generates collocation and boundary points within the unit square. All
    points are generated with random coordinates

    ----------------------------------------------------------------------------
    Arguments:
    num_Collocation_Points : The number of collocation points (within the domain)
    we should generate.

    num_Boundary_Points : Number of Boundary points we should generate. Thus
    MUST be a multiple of 4.

    ----------------------------------------------------------------------------
    Returns:
    A tuple of tensors. The first element is a (num_Collocation_Points)x2 tensor
    whose ith row holds the coordinates of the ith collocation point. The second
    element is a (num_Boundary_Points)x2 tensor whose ith row holds the
    coordinates of the ith boundary point. """

    assert((num_Boundary_Points % 4) == 0), "num_Boundary_Points must be a multiple of 4!"

    # Generate collocation points.
    Collocation_Points = torch.rand((num_Collocation_Points, 2));

    # We will generate boundary points for each of the four sides, as well as
    # the four corners. Therefore, each side will have num_Boundary_Points/4 - 1
    # (non-corner) points.
    num_Boundary_Points_per_side : int = num_Boundary_Points//4 - 1;

    # x coordinate is 0, y is random.
    Boundary_Points_Left =   torch.cat((torch.zeros((num_Boundary_Points_per_side, 1), dtype = torch.float),
                                        torch.rand((num_Boundary_Points_per_side, 1), dtype = torch.float)),
                                        dim = 1);

    # x coordinate is 1, y is random
    Boundary_Points_Right =  torch.cat((torch.ones((num_Boundary_Points_per_side, 1), dtype = torch.float),
                                        torch.rand((num_Boundary_Points_per_side, 1), dtype = torch.float)),
                                        dim = 1);

    # x coordinate is random, y is 0
    Boundary_Points_Bottom = torch.cat((torch.rand((num_Boundary_Points_per_side, 1), dtype = torch.float),
                                        torch.zeros((num_Boundary_Points_per_side, 1), dtype = torch.float)),
                                        dim = 1);

    # x coordinate is random, y is 1
    Boundary_Points_Top =    torch.cat((torch.rand((num_Boundary_Points_per_side, 1), dtype = torch.float),
                                        torch.ones((num_Boundary_Points_per_side, 1), dtype = torch.float)),
                                        dim = 1);

    Boundary_Corners = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype = torch.float);

    # Concatenate everything together!
    Boundary_Points = torch.cat((Boundary_Points_Left,
                                 Boundary_Points_Right,
                                 Boundary_Points_Bottom,
                                 Boundary_Points_Top,
                                 Boundary_Corners),
                                 dim = 0);

    return (Collocation_Points, Boundary_Points);



# main function!
def main():
    # Load setup data from the setup file.
    Setup_Data = Setup_File_Reader();

    # Test that we got the correct input.
    print("Training PINN with the following parameters:")
    for item in Setup_Data.__dict__.items():
        print(item);

    # Initialize Network hyperparameters.
    Epochs : int = Setup_Data.Epochs;
    Learning_Rate : float = Setup_Data.Learning_Rate;

    # Initalize training parameters
    Num_Train_Coloc_Points : int = Setup_Data.Num_Train_Coloc_Points;
    Num_Train_Bound_Points : int = Setup_Data.Num_Train_Bound_Points;
    Num_Test_Coloc_Points  : int = Setup_Data.Num_Test_Coloc_Points;
    Num_Test_Bound_Points  : int = Setup_Data.Num_Test_Bound_Points;

    # Set up the neural network to approximate the PDE solution.
    u_NN = Neural_Network(  Num_Hidden_Layers   = Setup_Data.Num_Hidden_Layers,
                            Nodes_Per_Layer     = Setup_Data.Nodes_Per_Layer,
                            Input_Dim           = 2,
                            Output_Dim          = 1);

    # Pick an optimizer.
    Optimizer = torch.optim.Adam(u_NN.parameters(), lr = Learning_Rate);

    # If we're loading from file, load in the store network's parameters.
    if(Setup_Data.Load_Network_State == True or Setup_Data.Load_Optimize_State == True):
        # Load the saved checkpoint.
        Checkpoint = torch.load(Setup_Data.Load_File_Name);

        if(Setup_Data.Load_Network_State == True):
            u_NN.load_state_dict(Checkpoint["Network_State"]);
            u_NN.train();

        # Note that this will overwrite the specified Learning Rate using the
        # Learning rate in the saved state. Thus, if this is set to true, then
        # we essentially ignore the learning rate in the setup file.
        if(Setup_Data.Load_Optimize_State == True):
            Optimizer.load_state_dict(Checkpoint["Optimizer_State"]);

    # Set up testing and training collocation/boundary points.
    (Training_Coloc_Points, Training_Bound_Points) = generate_points(num_Collocation_Points = Num_Train_Coloc_Points, num_Boundary_Points = Num_Train_Bound_Points);
    (Testing_Coloc_Points,  Testing_Bound_Points)  = generate_points(num_Collocation_Points = Num_Test_Coloc_Points, num_Boundary_Points = Num_Test_Bound_Points);

    # Set up array to hold the testing losses.
    Collocation_Losses = np.empty((Epochs), dtype = np.float);
    Boundary_Losses    = np.empty((Epochs), dtype = np.float);

    # Set up figure and Axes.
    fig, Axes = Setup_Axes();

    # Set up plotting gridpoints.
    Plotting_Points = Generate_Plot_Gridpoints(50);

    # Loop through the epochs.
    for t in range(Epochs):
        # Run training, testing for this epoch. Log the losses
        Training_Loop(  u_NN,
                        Collocation_Points = Training_Coloc_Points,
                        Boundary_Points   = Training_Bound_Points,
                        Optimizer = Optimizer);

        (Collocation_Losses[t], Boundary_Losses[t]) = Testing_Loop( u_NN,
                                                                    Collocation_Points = Testing_Coloc_Points,
                                                                    Boundary_Points = Testing_Bound_Points );

        # Print losses.
        print(("Epoch #%-4d: " % t), end = '');
        print(("\tCollocation Loss = %7f" % Collocation_Losses[t]), end = '');
        print((",\t Boundary Loss = %7f" % Boundary_Losses[t]), end = '');
        print((",\t Total Loss = %7f" % (Collocation_Losses[t] + Boundary_Losses[t])));

    # Save the network and optimizer states!
    if(Setup_Data.Save_To_File == True):
        torch.save({"Network_State" : u_NN.state_dict(),
                    "Optimizer_State" : Optimizer.state_dict()},
                    Setup_Data.Save_File_Name);

    # Plot final results.
    Update_Axes(fig, Axes, u_NN, Plotting_Points, 50);
    plt.show();


if __name__ == '__main__':
    main();
