import torch
import torch.nn as nn

class SimpleMLP(nn.Module):

    def __init__(self, in_size, h_sizes, out_size):

        super(SimpleMLP, self).__init__()

        # Hidden layers
        self.input = nn.Linear(in_size, h_sizes[0])

        # Hidden layers
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))

        # Output layer
        self.out = nn.Linear(h_sizes[-1], out_size)
        self.sigmoid = nn.Sigmoid()

        self.initialize_weights()

    def forward(self, x, mask):

        x = torch.relu(self.input(x))

        # Feedforward
        for layer in self.hidden:
            x = torch.relu(layer(x))
        x = self.out(x)
        output = self.sigmoid(x)
        output = output.squeeze(-1)
        return output, mask

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)