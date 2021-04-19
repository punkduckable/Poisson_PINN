import torch;
import numpy as np;



# The Neural Network class!
class Neural_Network(torch.nn.Module):
    def __init__(self,
                 num_hidden_layers : int = 3,    # Number of Hidden Layers
                 nodes_per_layer : int   = 20,   # Nodes in each Hidden Layer
                 input_dim : int         = 1,    # Number of components in the input
                 output_dim : int        = 1):   # Number of components in the output
        # Note: we assume that num_hidden_layers, nodes_per_layer, input_dim,
        # and out_dim are positive integers.
        assert (num_hidden_layers > 0   and
                nodes_per_layer > 0     and
                input_dim > 0           and
                output_dim > 0), "Neural_Network initialization arguments must be positive integers!"

        # Call the superclass initializer.
        super(Neural_Network, self).__init__();

        # Define object attributes. Note that there is an optput layer in
        # addition to the hidden layers (which is why Num_Layers is
        # num_hidden_layers + 1)
        self.input_dim : int  = input_dim;
        self.output_dim : int = output_dim;
        self.Num_Layers : int = num_hidden_layers + 1;

        # Define Layers ModuleList.
        self.Layers = torch.nn.ModuleList();

        # Append the first hidden layer. The domain of this layer is the input
        # domain, which means that in_features = input_dim. Since this is a
        # hidden layer, however it must have nodes_per_layer output features.
        self.Layers.append(
            torch.nn.Linear(    in_features  = input_dim,
                                out_features = nodes_per_layer,
                                bias = True )
        );

        # Now append the rest of the hidden layers. Each of these layers maps
        # within the same space, which means that in_features = out_features.
        # Note that we start at i = 1 because we already made the 1st
        # hidden layer.
        for i in range(1, num_hidden_layers):
            self.Layers.append(
                torch.nn.Linear(    in_features  = nodes_per_layer,
                                    out_features = nodes_per_layer,
                                    bias = True )
            );

        # Now, append the Output Layer, which has nodes_per_layer input
        # features, but only output_dim output features.
        self.Layers.append(
            torch.nn.Linear(    in_features  = nodes_per_layer,
                                out_features = output_dim,
                                bias = True )
        );

        # Initialize the weight matricies, bias vectors in the network.
        for i in range(self.Num_Layers):
            torch.nn.init.xavier_uniform_(self.Layers[i].weight);

        # Finally, set the Network's activation function.
        self.Activation_Function = torch.nn.Tanh();

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # Note: the input must be an input_dim dimensional (1d) tensor.

        # Pass x through the network's layers!
        for i in range(self.Num_Layers - 1):
            x = self.Activation_Function(self.Layers[i](x));

        # Pass through the last layer and return (there is no activation
        # function in the last layer)
        return self.Layers[self.Num_Layers - 1](x);



# Driving term for Poission's equation
def f(x : float, y : float) -> torch.Tensor:
    """ Poission's equation is -(d^2u/dx^2 + d^2u/dy^2) = f, for some function
    f. This function defines f.

    ----------------------------------------------------------------------------
    Arguments:
    x : x-coordinate we want to evaluate f at.
    y : y-coordinate we want to evaluate f at.

    ----------------------------------------------------------------------------
    Returns:
    A scalar (single element) tensor containing f(x, y). """

    return  (2 * (np.pi ** 2) *
            torch.sin(np.pi * x) *
            torch.sin(np.pi * y));



# Loss from enforcing the PDE at the colocation points.
def Colocation_Loss(u_NN : Neural_Network, Colocation_Points : torch.Tensor) -> torch.Tensor:
    """ This function evaluates how well u_NN satisifies the PDE at the
    colocation points. For brevity, let u = u_NN. At each colocation point,
    we compute the following:
                                d^2u/dx^2 + d^2u/dy^2 + f
    If u actually satisified the PDE, then this whould be zero everywhere.
    However, it generally won't be. This function computes the square of the
    quantity above at each Colocation point. We return the mean of these squared
    errors.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : The neural network that approximates the solution.

    colocation_points : a tensor of coordinates of the colocation points. If
    there are N colocation points, then this should be a N x 2 tensor, whose ith
    row holds the coordinates of the ith colocation point.

    ----------------------------------------------------------------------------
    Returns:
    Mean Square Error at the colocation points. """

    num_Colocation_Points : int = Colocation_Points.shape[0];

    # Now, initialize the loss and loop through the colocation points!
    Loss = torch.zeros(1, dtype = torch.float);
    for i in range(num_Colocation_Points):
        xy = Colocation_Points[i];

        # We need to compute the gradeint of u with respect to the xy coordinates.
        xy.requires_grad_(True);

        # Calculate approximate solution at this colocation point.
        u = u_NN.forward(xy);

        # Compute gradient of u with respect to xy. We have to create the graph
        # used to compute grad_u so that we can evaluate second derivatives.
        # We also need to set retain_graph to True (which is implicitly set by
        # setting create_graph = True, though I keep it to make the code more
        # explicit) so that torch keeps the computational graph for u, which we
        # will need when we do backpropigation.
        grad_u = torch.autograd.grad(u, xy, retain_graph = True, create_graph = True)[0];

        # compute du/dx and du/dy. grad_u is a two element tensor. It's first
        # element holds du/dx, and its second element holds du/dy.
        du_dx = grad_u[0];
        du_dy = grad_u[1];

        # Now compute the gradients of du_dx and du_dy with respect to xy. We
        # need to create graphs for both of these so that torch can track these
        # operations when constructing the computational graph for the loss
        # function (which it will use in backpropigation). We also need to
        # retain the graph for grad_u for when we do backpropigation.
        grad_du_dx = torch.autograd.grad(du_dx, xy, retain_graph = True, create_graph = True)[0];
        grad_du_dy = torch.autograd.grad(du_dy, xy, retain_graph = True, create_graph = True)[0];


        # We want the d^2u/dx^2 and d^2u/dy^2, which should be the [0] and
        # [1] elements of grad_du_dx and grad_du_dy, respectively.
        d2u_dx2 = grad_du_dx[0];
        d2u_dy2 = grad_du_dy[1];

        # Now evaluate the driving term of the PDE at the current point.
        Loss += (d2u_dx2 + d2u_dy2 + f(xy[0], xy[1])) ** 2;

    # Divide the accmulated loss by the number of colocation points to get
    # the mean square colocation loss.
    return (Loss / num_Colocation_Points);



# Boundary loss
def Boundary_Loss(u_NN : Neural_Network, Boundary_Points : torch.Tensor, c : float) -> torch.Tensor:
    """ This function imposes the Dirichlet boundary condition u = c.
    Specifically, for each boundary point (x,y), it computes the square of the
    difference between u(x,y) and c (the imposed BC). We return the mean of
    these squared errors.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : The neural network that approximates the solution.

    Boundary_Points : a tensor of coordinates of the colocation points. If
    there are N boundary points, then this should be a N x 2 tensor, whose ith
    row holds the coordinates of the ith boundary point.

    c : The BC that we're imposing (should be a constant).

    ----------------------------------------------------------------------------
    Returns:
    Mean Square Error at the boundary points. """

    num_Boundary_Points : int = Boundary_Points.shape[0];

    # Now, initialize the Loss and loop through the Boundary Points.
    Loss = torch.zeros(1, dtype = torch.float);
    for i in range(num_Boundary_Points):
        xy = Boundary_Points[i];

        # Compute approximate solution at this boundary point.
        u = u_NN.forward(xy);

        # Aggregate square of difference between the required BC and
        # the approximate solution at this boundary point.
        Loss += (u - c)**2;

    # Divide the accmulated loss by the number of boundary points to get
    # the mean square boundary loss.
    return (Loss / num_Boundary_Points);
