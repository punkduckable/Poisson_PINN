from Network import Network;

import torch;



def BC_Loss(U           : Network,
            BC_Coords   : torch.Tensor,
            BC_Targets  : torch.Tensor) -> torch.Tensor:
    """ This function imposes boundary conditions.

    ----------------------------------------------------------------------------
    Arguments:

    U : This is the neural network approximation to the system response function.

    BC_Coords : This is an N by 2 tensor, where N is the number of points at
    which we want to impose boundary conditions. The ith row of this tensor
    should hold the (x, y) coordinates of the ith point at which we will impose
    a boundary condition.

    BC_Targets : This is an N by 1 tensor, where N is the number of points at
    which we want to impose a boundary condition. The ith element of this tensor
    should hold the value we want to force the solution to assume at the ith
    boundary coordinate.

    ----------------------------------------------------------------------------
    Returns:

    A scalar (one-element) tensor whose lone entry holds the mean square error
    between U evaluated at the BC_Coords, and the target values at those coords. """

    # First, evaluate U at the coordinates. When you pass an N by 2 tensor
    # through a network that maps from R^2 to R, the output is an N by 1 matrix.
    # The result is an N by 1 tensor. We want an N element tensor. To do this,
    # we view it as a 1d tensor. To do this, we use the view method (which can
    # change the shape of a tensor without changing its contents).
    U_BC            : torch.Tensor = U(BC_Coords).view(-1)

    # Now, evaluate the square difference between U_BC and BC_Targets.
    Square_Errors   : torch.Tensor = torch.mul(torch.sub(U_BC, BC_Targets), torch.sub(U_BC, BC_Targets));

    # Return the mean square error!
    return torch.mean(Square_Errors);



def PDE_Residual(   U          : Network,
                    f          : torch.nn.Module,
                    Coords     : torch.Tensor) -> torch.Tensor:
    """ This function evaluates the PDE residual,
            D_x^2 U(x, y) + D_y^2 U(x, y) - f(x, y)
    For each (x, y) in Coords.

    ----------------------------------------------------------------------------
    Arguments:

    U : This is the neural network approximation to the system response function.

    f : This is the driving term in the Poission equation. It should be a
    torch.Module object which takes in N by 2 tensor of coordinates (ith row
    is the ith coordinate) and returns an N element tensor whose ith entry
    holds f evaluated at the ith row of the input.

    Coords : A N by 2 tensor whose ith row holds the (x, y) coordinates of the
    ith collocation point.

    ----------------------------------------------------------------------------
    Returns:

    A N by 1 tensor whose ith entry holds the PDE residual at the ith coordinate. """

    # First, make sure that Coords requires grad. Without this, we can not
    # differentiate U with respect to its input coordinates. In particular,
    # if (x, y) has requires_grad_ = True, then we can differentiate U(x, y)
    # with respect to x and y.
    Coords.requires_grad_(True);

    # Next, evaluate U at the coordinates.
    U_Coords : torch.Tensor = U(Coords).view(-1);

    # Next, differentiate U with respect to its input coordinates. This requires
    # use of a "grad_outputs" variable. The details of what this is doing are
    # fairly complex (ask me if you want more details). The important thing to
    # remember is that if you set it to a vector of ones whose size matches that
    # of U_Coords, then the (i, j) entry of the resulting gradient holds the
    # derivative of U with respect to its jth input, evaluated at the ith
    # Coordinate.
    Grad_U : torch.Tensor = torch.autograd.grad(outputs         = U_Coords,
                                                inputs          = Coords,
                                                grad_outputs    = torch.ones_like(U_Coords),
                                                retain_graph    = True,
                                                create_graph    = True)[0];

    # You may be wondering... what do retain_graph and create_graph do? When you
    # evaluate U at the Coords, pytorch constructs a computational graph that
    # tracks the computations required to get from Coords to U_Coords. Torch
    # uses this graph to perform back-backpropigation. By default, however, when
    # we autodiff, torch frees that graph. This is bad, because we need the
    # graph to still exist for when we perform back-propigation. retain_graph
    # disables this "clean up" behavior. On the other hand, to back-propigate
    # through the gradient, torch needs to know how to get from Coords to
    # Grad_U. To do this, it needs to construct a computational graph for the
    # gradient evaluation. create_graph tells torch to construct that graph.

    Dx_U : torch.Tensor = Grad_U[:, 0].view(-1);
    Dy_U : torch.Tensor = Grad_U[:, 1].view(-1);

    # Now, differentiate Dx_U and Dy_U with respect to the input coordinates
    # to evaluate Dx2_U, Dy2_U.
    Grad_Dx_U : torch.Tensor = torch.autograd.grad( outputs         = Dx_U,
                                                    inputs          = Coords,
                                                    grad_outputs    = torch.ones_like(Dx_U),
                                                    retain_graph    = True,
                                                    create_graph    = True)[0];
    Dx2_U : torch.Tensor = Grad_Dx_U[:, 0];

    Grad_Dy_U : torch.Tensor = torch.autograd.grad( outputs         = Dy_U,
                                                    inputs          = Coords,
                                                    grad_outputs    = torch.ones_like(Dy_U),
                                                    retain_graph    = True,
                                                    create_graph    = True)[0];
    Dy2_U : torch.Tensor = Grad_Dy_U[:, 1];

    # Finally, evaluate f at the coordinates.
    f_Coords = f(Coords);

    # Finally, evaluate the PDE residual at each coordinate.
    Residual_Coords = torch.sub(torch.add(Dx2_U, Dy2_U), f_Coords);

    # We could have also used operator overloading:
    #   Residual_Coords = Dx2_U + Dy2_U - f_Coords
    # this is equivalent to the line above. Most math operations in torch
    # (including pow, add, sub) act element-wise.

    return Residual_Coords;



def Coll_Loss(  U          : Network,
                f          : torch.nn.Module,
                Coords     : torch.Tensor) -> torch.Tensor:
    """ This function forces U to satisify the Poission equation:
            D_x^2 U(x, y) + D_y^2 U(x, y) = f(x, y)
    For each (x, y) in Coords.

    ----------------------------------------------------------------------------
    Arguments:

    U : This is the neural network approximation to the system response function.

    f : This is the driving term in the Poission equation. It should be a
    torch.Module object which takes in N by 2 tensor of coordinates (ith row
    is the ith coordinate) and returns an N element tensor whose ith entry
    holds f evaluated at the ith row of the input.

    Coords : A N by 2 tensor whose ith row holds the (x, y) coordinates of the
    ith collocation point.

    ----------------------------------------------------------------------------
    Returns:

    A scalar (one-element) tensor whose lone entry holds the mean square error
    of the PDE residual, D_x^2 U - D_y^2 U - f, evaluated at the Coordinates. """

    # Ealuate the PDE residual at each coordinate.
    Residual_Coords = PDE_Residual(     U       = U,
                                        f       = f,
                                        Coords  = Coords);

    # We could have also used operator overloading:
    #   Residual_Coords = Dx2_U + Dy2_U - f_Coords
    # this is equivalent to the line above. Most math operations in torch
    # (including pow, add, sub) act element-wise.

    # Now, return the mean square PDE residual.
    return torch.mean(torch.pow(Residual_Coords, 2));
