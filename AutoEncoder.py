import os
import numpy as np
import torch
import torchvision
from torch import nn

class autoencoder(nn.Module):
    def __init__(self, dims):
        super(autoencoder, self).__init__()
        self.num_layers = len(dims)-1
        self.encoder = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(self.num_layers)])
        self.decoder = nn.ModuleList([nn.Linear(dims[i], dims[i - 1]) for i in range(self.num_layers,0,-1)])
        # self.encoder = nn.Sequential(
        #     nn.Linear(dims[0], dims[1]),
        #     nn.ReLU(True),
        #     nn.Linear(dims[1], dims[2]),
        #     nn.ReLU(True), nn.Linear(dims[2], dims[3]))
        # self.decoder = nn.Sequential(
        #     nn.Linear(dims[3], dims[2]),
        #     nn.ReLU(True),
        #     nn.Linear(dims[2], dims[1]),
        #     nn.ReLU(True),
        #     nn.Linear(dims[1], dims[0]),
        #     nn.Tanh())

    def encode(self, x):
        for i in range(self.num_layers-1):
            x = self.encoder[i](x)
            x = nn.functional.relu(x)
        x = self.encoder[self.num_layers-1](x)
        return x

    def decode(self, x):
        for i in range(self.num_layers-1):
            x = self.decoder[i](x)
            x = nn.functional.relu(x)
        x = self.decoder[self.num_layers-1](x)
        x = nn.functional.tanh(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    dim = 28 ** 2
    # dims = [dim] * 4
    dims = [dim,500,100,50]
    x = torch.Tensor(np.random.rand(5, dim))
    net = autoencoder(dims)
    bottle_neck = net.encode(x)
    out = net.decode(bottle_neck)
    a = 3
