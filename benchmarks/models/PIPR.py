# PIPR is implemented based on the original Keras implementation.
# Ref: https://github.com/muhaochen/seq_ppi/blob/master/binary/model/lasagna/rcnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PIPR(nn.Module):
    def __init__(self, hidden_dim=50):
        super().__init__()
        self.conv_1 = nn.Conv1d(12, hidden_dim, kernel_size=3)
        self.gru_1 = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True
        )
        self.conv_2 = nn.Conv1d(150, hidden_dim, kernel_size=3)
        self.gru_2 = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True
        )
        self.conv_3 = nn.Conv1d(150, hidden_dim, kernel_size=3)
        self.gru_3 = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True
        )
        self.conv_4 = nn.Conv1d(150, hidden_dim, kernel_size=3)

        self.fc = nn.Sequential(
            nn.Linear(50, 100),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(100, int((hidden_dim + 7) / 2)),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(int((hidden_dim + 7) / 2), 1),
        )

    def forward(self, x_ab, x_ag):
        x_ab = self.conv_1(x_ab)
        x_ab = F.max_pool1d(x_ab, 3)
        x_ab_gru, _ = self.gru_1(x_ab.permute(0, 2, 1))
        x_ab = torch.cat([x_ab_gru.permute(0, 2, 1), x_ab], axis=1)
        x_ab = self.conv_2(x_ab)
        x_ab = F.max_pool1d(x_ab, 3)
        x_ab_gru, _ = self.gru_2(x_ab.permute(0, 2, 1))
        x_ab = torch.cat([x_ab_gru.permute(0, 2, 1), x_ab], axis=1)
        x_ab = self.conv_3(x_ab)
        x_ab = F.max_pool1d(x_ab, 3)
        x_ab_gru, _ = self.gru_3(x_ab.permute(0, 2, 1))
        x_ab = torch.cat([x_ab_gru.permute(0, 2, 1), x_ab], axis=1)
        x_ab = self.conv_4(x_ab)
        x_ab = torch.mean(x_ab, dim=2)

        x_ag = self.conv_1(x_ag)
        x_ag = F.max_pool1d(x_ag, 3)
        x_ag_gru, _ = self.gru_1(x_ag.permute(0, 2, 1))
        x_ag = torch.cat([x_ag_gru.permute(0, 2, 1), x_ag], axis=1)
        x_ag = self.conv_2(x_ag)
        x_ag = F.max_pool1d(x_ag, 3)
        x_ag_gru, _ = self.gru_2(x_ag.permute(0, 2, 1))
        x_ag = torch.cat([x_ag_gru.permute(0, 2, 1), x_ag], axis=1)
        x_ag = self.conv_3(x_ag)
        x_ag = F.max_pool1d(x_ag, 3)
        x_ag_gru, _ = self.gru_3(x_ag.permute(0, 2, 1))
        x_ag = torch.cat([x_ag_gru.permute(0, 2, 1), x_ag], axis=1)
        x_ag = self.conv_4(x_ag)
        x_ag = torch.mean(x_ag, dim=2)

        output = self.fc(torch.multiply(x_ab, x_ag))
        return output
