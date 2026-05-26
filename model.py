import torch
import torch.nn as nn

class ChessNet(nn.Module):

    def __init__(self, num_moves):

        super(ChessNet, self).__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(

            nn.Linear(8 * 8, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, num_moves)

        )

    def forward(self, x):

        x = self.flatten(x)

        return self.network(x)
    