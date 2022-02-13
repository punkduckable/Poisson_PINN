import torch;


# A network is a map on tensors (it maps a tensor in R^n to a tensor in R^m). As
# such, it is a Module (recall, a Module is a function of a tensors). Thus, we
# must subclass the Module class. All Module objects must have two methods
# defined, __init__ and forward. __init__ is just an initializer, while forward
# defines how the module maps input tensors to their image (a module is a
# function of a tensor!!!)
class Network(torch.nn.Module):
    def __init__(   self,
                    num_hidden_layers : int,
                    units_per_layer   : int) -> None:
        # First, call the Module constructor.
        super().__init__()

        # Second, set some class attributes
        self.num_hidden_layers : int = num_hidden_layers;
        self.units_per_layer   : int = units_per_layer;

        # A ModuleList is essentially a list of Module objects. You can
        # Modules to it using the append method, and access its members using
        # square brackets.
        self.Layers : torch.nn.Module = torch.nn.ModuleList();

        # Set up the first hidden layer. A linear object (confusingly)
        # implements an affine map x -> xW^T + b from R^n to R^m. Here, x is a
        # B by n matrix whose ith row holds the ith input, W is an m by n matrix
        # of weights, and b is a m by 1 vector of biases.
        self.Layers.append(torch.nn.Linear( in_features     = 2,    # x, y
                                            out_features    = units_per_layer,
                                            dtype           = torch.float32));

        # Set up the other hidden layers
        for i in range(1, num_hidden_layers):
            self.Layers.append(torch.nn.Linear(in_features  = units_per_layer,
                                               out_features = units_per_layer,
                                               dtype        = torch.float32));

        # Set up the final layer
        self.Layers.append(torch.nn.Linear( in_features     = units_per_layer,
                                            out_features    = 1,
                                            dtype           = torch.float32));

        # Now, initialize the weight matrices and bias vectors. Recall that a
        # Linear object, L, implements an affine map, x -> xW^T + b. W and b are
        # tensor attributes of L. In particular, W = L.weight, b = L.bias. We
        # will initialize W using the Xavier normal distribution, and zero
        # initialize b.
        for i in range(num_hidden_layers + 1):
            torch.nn.init.xavier_normal_(   self.Layers[i].weight);
            torch.nn.init.zeros_(           self.Layers[i].bias);

        # Finally, set an activation function.
        self.Activation_Function = torch.nn.Tanh();

    def forward(self, X : torch.Tensor) -> torch.Tensor:
        # X should be a B by 2 tensor whose ith row holds the (x, y) components
        # of the ith component we want to evaluate the network at.

        # Pass X through the layers!
        for i in range(self.num_hidden_layers):
            X = self.Activation_Function(self.Layers[i](X));

        # No activation function on the last layer!
        return self.Layers[self.num_hidden_layers](X);
