import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, num_classes=2) -> None:
        super(Net, self).__init__()
        self.l1 = nn.Linear(30, 16)
        self.l2 = nn.Linear(16, 4)
        self.l3 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        # output layer
        x = self.sigmoid(self.l3(out2))
        return x
