import numpy as np
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
from models.loc_rnn_layer import *
from models.convgru_layer import ConvGRU
from models.hgru_layer import hConvGRU


class RNNSegm(nn.Module):
    def __init__(self, rnn_name, depth, width=2):
        super().__init__()
        self.rnn_name = rnn_name
        self.inplanes = depth
        print(self.inplanes)
        self.width = 2  # hard coded, following ResidualNetworkSegment style of model
        self.inplanes = self.inplanes * self.width

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        if self.rnn_name.startswith("locrnn"):
            self.rnn = LocRNNLayer(self.inplanes, self.inplanes, 
                                    3, 5, 3, timesteps=15)
        elif self.rnn_name.startswith("hgru"):
            self.rnn = hConvGRU(
                filt_size=5,  # do not change, matching DaleRNNLayer
                hidden_dim=self.inplanes,
                timesteps=15)

        elif self.rnn_name.startswith("gru"):
            self.rnn = ConvGRU(self.inplanes, self.inplanes, 5, 15)
        else:
            raise NotImplementedError("RNN name not recognized")

        self.conv2 = nn.Conv2d(self.inplanes, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 2, kernel_size=3,
                               stride=1, padding=1, bias=False)

    def forward(self, x):  
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.rnn(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x

# width = 2 is fixed in the class defined above, as the experiment used width = 2 across all models
# width at the functions below are reserved for loc_rnn_eval.py and used as timestep
def locrnn(num_outputs, depth, width, dataset):
    return RNNSegm("locrnn", depth)

def hgru(num_outputs, depth, width, dataset):
    return RNNSegm("hgru", depth)

def gru(num_outputs, depth, width, dataset):
    return RNNSegm("gru", depth)