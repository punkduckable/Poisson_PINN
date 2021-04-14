import torch;
import numpy as np;
import torchviz;

# The Neural Network class!
class Neural_Network(torch.nn.Module):
    def __init__(self,
                num_hidden_layers : int = 3,     # Number of Hidden Layers
                nodes_per_layer : int   = 20,    # Nodes in each Hidden Layer
                input_dim : int         = 1,     # Number of components in the input
                output_dim : int        = 1):    # Number of components in the output
        # Note: we assume that num_hidden_layers, nodes_per_layer, input_dim,
        # and out_dim are positive integers.
        assert (num_hidden_layers > 0 and
                nodes_per_layer > 0 and
                input_dim > 0 and
                output_dim > 0), "Neural_Network initialization arguments must be positive integers!"

        # Call the superclass initializer.
        super(Neural_Network, self).__init__();

        # Define object attributes.
        self.input_dim  = input_dim;
        self.output_dim = output_dim;
        self.Num_Layers = num_hidden_layers + 1;

        # Define Layers ModuleList.
        self.Layers = torch.nn.ModuleList();

        # Append the first hidden layer. The domain of this layer is the input
        # domain, which means that in_features = input_dim. Since this is a
        # hidden layer, however it must have nodes_per_layer output features.
        self.Layers.append(
            torch.nn.Linear(
                in_features  = input_dim,
                out_features = nodes_per_layer,
                bias = True
            )
        );

        # Now append the rest of the hidden layers. Each of these layers maps
        # within the same space, which means thatin_features = out_features.
        for i in range(1, num_hidden_layers):
            self.Layers.append(
                torch.nn.Linear(
                    in_features  = nodes_per_layer,
                    out_features = nodes_per_layer,
                    bias = True
                )
            );

        # Now, append the Output Layer, which has nodes_per_layer input
        # features, but only output_dim output features.
        self.Layers.append(
            torch.nn.Linear(
                in_features  = nodes_per_layer,
                out_features = output_dim,
                bias = True
            )
        );

        # Set the number of layers
        self.Num_Layers = num_hidden_layers + 1;

        # Initialize the weight matricies in the network.
        for i in range(self.Num_Layers):
            torch.nn.init.xavier_uniform_(self.Layers[i].weight);

        # Finally, set the Network's activation function.
        self.Activation_Function = torch.nn.Tanh();

    def forward(self, x):
        # Note: the input must be an input_dim-component tensor.

        # Pass x through the network's layers!
        for i in range(self.Num_Layers - 1):
            x = self.Activation_Function(self.Layers[i](x));

        # Pass through the last layer and return (no activation function)
        return self.Layers[self.Num_Layers - 1](x);



# Poission's equation is -(d^2u/dx^2 + d^2u/dy^2) = f, for some function f.
# This function defines f.
def f(x : float, y : float) -> torch.Tensor:
    """ Arguments:
    x : x-coordinate we want to evaluate f at.
    y : y-coordinate we want to evaluate f at.

    Returns:
    A scalar tensor containing f(x, y). """

    return  (2 * (np.pi ** 2) *
            torch.sin(np.pi * x) *
            torch.sin(np.pi * y));



# True solution to Poission's equation with the above driving term.
def true_solution(x : float, y : float) -> torch.Tensor:
    """ Arguments:
    x : x-coordinate we want to evaluate the true solution at.
    y : y-coordinate we want to evaluate the true solution at.

    Returns:
    A scalar tensor containing the true solution at (x, y). """

    return np.sin(np.pi * x) * np.sin(np.pi * y);



# Loss from enforcing the PDE at the colocation points.
def Colocation_Loss(net : Neural_Network, colocation_points : torch.Tensor) -> torch.Tensor:
    """ Arguments:
    net : The neural network that approximates the solution.
    colocation_points : a tensor of coordinates of the colocation points. If
    there are N colocation points, then this should be a N x 2 tensor, whose ith
    row holds the coordinates of the ith colocation point.

    Returns:
    Mean Square Error at the colocation points.
    """

    # First, determine number of colocation points (rows of this arg)
    num_colocation_points : int = colocation_points.shape[0];

    # Now, initialize the loss and loop through the colocation points!
    Loss = torch.zeros(1, dtype = torch.float);
    for i in range(num_colocation_points):
        xy = colocation_points[i];

        # We need to compute the gradeint of u with respect to the xy coordinates.
        xy.requires_grad_(True);

        # Calculate approximate solution at this colocation point.
        u = net.forward(xy);

        # Compute gradient of u with respect to xy. We have to create the graph
        # used to compute grad_u so that we can evaluate second derivatives.
        # This also implicitly sets retain_graph = True.
        grad_u = torch.autograd.grad(u, xy, create_graph = True)[0];

        # compute du/dx and du/dy. grad_u is a two element tensor. It's first
        # element holds du/dx, and its second element holds du/dy.
        u_x = grad_u[0];
        u_y = grad_u[1];

        # Now compute the gradients of u_x and u_y with respect to xy. We need
        # to create graphs for both of these so that torch can track these
        # operations when constructing the computational graph for the loss
        # function (which it will use in backpropigation)
        grad_u_x = torch.autograd.grad(u_x, xy, create_graph = True)[0];
        grad_u_y = torch.autograd.grad(u_y, xy, create_graph = True)[0];


        # We want the d^2u/dx^2 and d^2u/dy^2, which should be the [0] and
        # [1] elements of grad_u_x and grad_u_y, respectively.
        u_xx = grad_u_x[0];
        u_yy = grad_u_y[1];

        # Now evaluate the driving term of the PDE at the current point.
        Loss += (u_xx + u_yy + f(xy[0], xy[1])) ** 2;


    # Divide the accmulated loss by the number of colocation points to get
    # the mean square colocation loss.
    return Loss / num_colocation_points;



# Boundary loss
def Boundary_Loss(net: Neural_Network, Boundary_Points : torch.Tensor) -> torch.Tensor:
    """ Arguments:
    net : The neural network that approximates the solution.
    Boundary_Points : a tensor of coordinates of the colocation points. If
    there are N boundary points, then this should be a N x 2 tensor, whose ith
    row holds the coordinates of the ith boundary point.

    Returns:
    Mean Square Error at the boundary points.
    """

    # First, determine the number of boundary points.
    num_boundary_points : int = Boundary_Points.shape[0];

    # Now, initialize the Loss and loop through the Boundary Points.
    Loss = torch.zeros(1, dtype = torch.float);
    for i in range(num_boundary_points):
        xy = Boundary_Points[i];

        # Compute approximate solution at this boundary point.
        u = net.forward(xy);

        # Aggregate square of difference between the required BC and
        # the approximate solution at this boundary point.
        Loss += (u - 0)**2;

    # Divide the accmulated loss by the number of boundary points to get
    # the mean square boundary loss.
    return Loss / num_boundary_points;



u_NN = Neural_Network(
    num_hidden_layers = 2,
    nodes_per_layer = 5,
    input_dim = 2,
    output_dim = 1);

points = torch.rand((10, 2), dtype = torch.float);
My_Loss_coloc = Colocation_Loss(u_NN, points);
My_Loss_bound = Boundary_Loss(u_NN, points);


print(My_Loss_coloc);
print(My_Loss_bound);
