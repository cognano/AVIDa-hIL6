import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(8000, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x_ab, x_ag):
        x_ab = self.flatten(x_ab)
        x_ag = self.flatten(x_ag)
        output = self.fc(torch.cat([x_ab, x_ag], axis=1))
        return output
