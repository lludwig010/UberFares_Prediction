import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_prob, seed_value=None):
        """
        Initialize MLP.
        """
        super(MLP, self).__init__()

        if seed_value is not None:
            nn.manual_seed(seed_value)

        # List to store the layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_prob))

        # Define Hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob)) 
        
                
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        #unpack the layers 
        self.model = nn.Sequential(*layers)

        self._initialize_weights()
    

    def _initialize_weights(self):
        """
        Initialize the weights and biases of the model.
        """

        #go through each layer
        for layer in self.model:
          #check if not relu
          if isinstance(layer, nn.Linear):
            #use xavier/glorot initialization for the weights
            nn.init.xavier_uniform_(layer.weight)
            #initialize bias to 0
            nn.init.zeros_(layer.bias)
    

    def forward(self, x):
        input = x
        for layer in self.model:
          y = layer(input)
          input = y

        return input
        
