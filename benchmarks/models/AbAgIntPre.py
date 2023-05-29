# AbAgIntPre is implemented based on the original implementation.
# Ref: https://github.com/emersON106/AbAgIntPre/blob/main/model/CNN.py
import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class AbAgIntPre(nn.Module):
    def __init__(self):
        super(AbAgIntPre, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=10, kernel_size=3, stride=1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            nn.Conv2d(10, 20, 3, 1),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(),
            Flatten(),
        )

        self.fc = nn.Sequential(nn.Linear(10240, 64), nn.Linear(64, 1))

    def forward_once(self, x):
        output = self.cnn(x)
        return output

    def forward(self, x_ab, x_ag):
        output1 = self.forward_once(x_ab)
        output2 = self.forward_once(x_ag)
        output = torch.cat((output1, output2), 1)
        output = self.fc(output)
        return output
