from Network import Network;
from Losses import Coll_Loss, BC_Loss;

import torch;



def Train(  U           : Network,
            f           : torch.nn.Module,
            Coll_Coords : torch.Tensor,
            BC_Coords   : torch.Tensor,
            BC_Targets  : torch.Tensor,
            Optim       : torch.optim.Optimizer) -> None:
    # This function performs one training epoch.

    # First, put U in train mode.
    U.train();

    # Next, evaluate the losses.
    Loss : torch.Tensor = BC_Loss(U, BC_Coords, BC_Targets) + Coll_Loss(U, f, Coll_Coords);

    # Next, zero out the gradients.
    Optim.zero_grad();

    # Now, run backpropigation.
    Loss.backward();

    # Finally, update the network parameters.
    Optim.step();

    # All done!
    return;



def Test(   U           : Network,
            f           : torch.nn.Module,
            Coll_Coords : torch.Tensor,
            BC_Coords   : torch.Tensor,
            BC_Targets   : torch.Tensor) -> None:
    # This function evaluates and reports the losses.

    # First, put U in evaluation mode.
    U.eval();

    # Next, evaluate the losses.
    Loss_BC   : torch.Tensor    = BC_Loss(U, BC_Coords, BC_Targets);
    Loss_Coll : torch.Tensor    = Coll_Loss(U, f, Coll_Coords);

    # report the losses! If T is a scalar (single-element) tensor, then T.item()
    # return T's lone element as a scalar (built-in) variable.
    print("BC = %9.6f, Coll = %9.6f, Total = %9.6f" % (Loss_BC.item(), Loss_Coll.item(), Loss_BC.item() + Loss_Coll.item()));

    # All done!
    return;
