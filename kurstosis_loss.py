import torch
import numpy as np


class KurtosisLoss(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(KurtosisLoss, self).__init__()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = torch.pow(x, 4) - 3
        x = torch.abs(x)
        return -torch.sum(x)


if __name__ == '__main__':
    x = torch.Tensor(np.random.rand(5, 10))
    Criterion = KurtosisLoss()
    loss = Criterion(x)
