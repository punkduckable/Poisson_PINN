import torch;
import numpy as np;
from typing import Tuple;
from Poisson_PINN import Neural_Network, f, True_Solution, Colocation_Loss, Boundary_Loss, PDE_Residual;


# Training Loop
def Training_Loop(  u_NN : Neural_Network,
                    Colocation_Points : torch.Tensor,
                    Boundary_Points : torch.Tensor,
                    optimizer : torch.optim.Optimizer) -> None:
    """ This loop runs one epoch of training for the neural network. In
    particular, we enforce the PDE at the specified Colocation_Points, and the
    boundary conditions at the Boundary_Points.

    ----------------------------------------------------------------------------
    Arguments:
    net : The neural network that is approximating the solution to the PDE.

    Colocation_Points : the colocation points at which we want to enforce the
    PDE. These should be on the interior of the domain. Futher, these should be
    DISTINCT from the points we test the network at. This should be an Nx2
    tensor of floats, where N is the number of colocation points. The ith
    row of this tensor should be the coordinates of the ith colocation point.

    Boundary_Points : The points on the boundary at which we want to enforce the
    boundary conditions. This should be a Bx2 tensor of floats, where B is
    the number of boundary points. The ith row of this tensor should hold the
    coordinates of the ith boundary point.

    optimizer : the optimizer we use to train u_NN.

    ----------------------------------------------------------------------------
    returns:
    Nothing! """

    # First, determine the number of Colocation and Boundary points.
    num_Colocation_Points = Colocation_Points.shape[0];
    num_Boundary_Points = Boundary_Points.shape[0];

    # Zero out the gradients in the neural network.
    optimizer.zero_grad();

    # Evaluate the Loss (Note, we enforce a BC of 0)
    Loss = (Colocation_Loss(u_NN, Colocation_Points) +
            Boundary_Loss(u_NN, Boundary_Points, 0));

    # Back propigate to compute gradients.
    Loss.backward();

    # update weights using optimizer.
    optimizer.step();



# Testing Loop
def Testing_Loop(   u_NN : Neural_Network,
                    Colocation_Points : torch.Tensor,
                    Boundary_Points : torch.Tensor) -> Tuple[float, float]:
    """ This loop tests the neural network at the specified Boundary and
    Colocation points. You CAN NOT run this funcetion with no_grad set True.
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
    net : The neural network that is approximating the solution to the PDE.

    Colocation_Points : the colocation points at which we want to enforce the
    PDE. These should be on the interior of the domain. Futher, these should be
    DISTINCT from the colocation points that we use in the training loop.
    This should be an Nx2 tensor of floats, where N is the number of colocation
    points. The ith row of this tensor should be the coordinates of the ith
    colocation point.

    Boundary_Points : The points on the boundary at which we want to enforce the
    boundary conditions. These should be DISTINCT from the boundary points we
    use in the training loop. This should be a Bx2 tensor of floats, where B is
    the number of boundary points. The ith row of this tensor should hold the
    coordinates of the ith boundary point.

    ----------------------------------------------------------------------------
    Returns:
    a tuple of floats. The first element holds the colocation loss, while
    the second holds the boundary loss. """

    Coloc_Loss = Colocation_Loss(u_NN, Colocation_Points).item();
    Bound_Loss = Boundary_Loss(u_NN, Boundary_Points, 0).item();

    # Should we worry
    return (Coloc_Loss, Bound_Loss);



# Generate colocation and boundary points
def generate_points(num_Colocation_Points : int, num_Boundary_Points : int) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Generates colocation and boundary points within the unit square. All
    points are generated with random coordinates

    ----------------------------------------------------------------------------
    Arguments:

    num_Colocation_Points : The number of colocation points (within the domain)
    we should generate.

    num_Boundary_Points : Number of Boundary points we should generate. Thus
    MUST be a multiple of 4.

    ----------------------------------------------------------------------------
    Returns:
    A tuple of tensors. The first element is a (num_Colocation_Points)x2 tensor
    whose ith row holds the coordinates of the ith colocation point. The second
    element is a (num_Boundary_Points)x2 tensor whose ith row holds the
    coordinates of the ith boundary point. """

    assert((num_Boundary_Points % 4) == 0), "num_Boundary_Points must be a multiple of 4!"

    # Generate colocation points.
    Colocation_Points = torch.rand((num_Colocation_Points, 2));

    # We will generate boundary points for each of the four sides, as well as
    # the four corners. Each side will get num_Boundary_Points/4 - 1 points.
    num_Boundary_Points_per_side = num_Boundary_Points//4 - 1;

    # x coordinate is 0, y is random.
    Boundary_Points_Left  =  torch.cat((torch.zeros((num_Boundary_Points_per_side, 1), dtype = torch.float),
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

    # all done!
    return (Colocation_Points, Boundary_Points);



# main function!
def main():
    # Specify hyperparameters
    Epochs : int = 10;
    Learning_Rate : float = .001;

    # Set up the neural network to approximate the PDE solution.
    u_NN = Neural_Network(
        num_hidden_layers = 5,
        nodes_per_layer = 20,
        input_dim = 2,
        output_dim = 1);

    # Pick an optimizer.
    Optimizer = torch.optim.Adam(u_NN.parameters(), lr = Learning_Rate);

    # Set up training and training colocation/boundary points.
    Training_Colocation_Points, Training_Boundary_Points = generate_points(num_Colocation_Points = 500, num_Boundary_Points = 400);
    Testing_Colocation_Points,  Testing_Boundary_Points  = generate_points(num_Colocation_Points = 100, num_Boundary_Points = 80);

    # Set up array to hold the testing losses.
    Colocation_Losses = np.empty((Epochs), dtype = np.float);
    Boundary_Losses   = np.empty((Epochs), dtype = np.float);

    # Loop through the epochs.
    for t in range(Epochs):
        # Run training, testing for this epoch. Log the losses
        Training_Loop(  u_NN,
                        Colocation_Points = Training_Colocation_Points,
                        Boundary_Points   = Training_Boundary_Points ,
                        optimizer = Optimizer);

        (Colocation_Losses[t], Boundary_Losses[t]) = Testing_Loop( u_NN,
                                                                   Colocation_Points = Testing_Colocation_Points,
                                                                   Boundary_Points = Testing_Boundary_Points );

        # Print losses.
        print(("Epoch #%d: " % t), end = '');
        print(("Colocation Loss = %7f" % Colocation_Losses[t]), end = '');
        print((", Boundary Loss = %7f" % Boundary_Losses[t]), end = '');
        print((", Total Loss = %7f" % (Colocation_Losses[t] + Boundary_Losses[t])));

if __name__ == '__main__':
    main();
