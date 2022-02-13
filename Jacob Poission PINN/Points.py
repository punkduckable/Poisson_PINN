import torch;
import numpy;
from typing import Tuple;



# Function to generate coordinates. We will use this to generate collocation
# and boundary points.
def Generate_Coords(    x_l         : float,
                        x_h         : float,
                        y_l         : float,
                        y_h         : float,
                        num_coords  : int) -> torch.Tensor:
    # First, generate X coordinates. torch.rand generates a tensor whose
    # elements are uniformly sampled from [0, 1). By scaling the output by
    # (b - a) and then adding a, we get a uniform distribution on [a, b).
    X = x_l + (x_h - x_l)*torch.rand((num_coords, 1), dtype = torch.float32);
    Y = y_l + (y_h - y_l)*torch.rand((num_coords, 1), dtype = torch.float32);

    # The cat function concatenates two or more tensors along a specified
    # dimension. This only works if the tensors' other dimensions have the
    # same size. We use this to combine X and Y, thereby generating a list of
    # (x, y) coordinates.
    Coords = torch.cat((X, Y), dim = 1);

    return Coords;



# Function to generate the boundary coordinates + target values at those
# coordinates. Feel free to change the four boundary functions and see what
# happens.
def Generate_Boundary(  x_l         : float,
                        x_h         : float,
                        y_l         : float,
                        y_h         : float,
                        num_coords  : int) -> Tuple[torch.Tensor, torch.Tensor]:
    # lower x boundary.
    x_low_coords = Generate_Coords(     x_l = x_l,
                                        x_h = x_l, # All coords have x = x_l!!!
                                        y_l = y_l,
                                        y_h = y_h,
                                        num_coords = num_coords);
    x_low_targets = torch.cos(numpy.pi * x_low_coords[:, 1]);


    # upper x boundary.
    x_high_coords = Generate_Coords(    x_l = x_h, # All coords have x = x_h!!!
                                        x_h = x_h,
                                        y_l = y_l,
                                        y_h = y_h,
                                        num_coords = num_coords);
    x_high_targets = torch.cos(numpy.pi * x_high_coords[:, 1]);

    # lower y boundary.
    y_low_coords = Generate_Coords(     x_l = x_l,
                                        x_h = x_h,
                                        y_l = y_l,
                                        y_h = y_l, # All coords have y = y_l!!!
                                        num_coords = num_coords);
    y_low_targets = torch.full_like(y_low_coords[:, 0], -1);

    # upper y boundary.
    y_high_coords = Generate_Coords(    x_l = x_l,
                                        x_h = x_h,
                                        y_l = y_h, # All coords have y = y_h!!!
                                        y_h = y_h,
                                        num_coords = num_coords);
    y_high_targets = torch.full_like(y_high_coords[:, 0], -1);

    # Now, concatenate the coordinates, targets together!
    BC_Coords = torch.cat(( x_low_coords,
                            x_high_coords,
                            y_low_coords,
                            y_high_coords),
                          dim = 0);

    BC_Targets = torch.cat((x_low_targets,
                            x_high_targets,
                            y_low_targets,
                            y_high_targets),
                          dim = 0);

    return (BC_Coords, BC_Targets);
