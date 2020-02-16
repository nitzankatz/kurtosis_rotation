import torch
import numpy as np

def symsqrt(matrix):
    """Compute the square root of a positive definite matrix."""
    # perform the decomposition
    # s, v = matrix.symeig(eigenvectors=True)
    _, s, v = matrix.svd()  # passes torch.autograd.gradcheck()
    _ , snp, vnp = np.linalg.svd(matrix.detach().numpy())


    # truncate small components
    above_cutoff = s > s.max() * s.size(-1) * torch.finfo(s.dtype).eps
    x_where, y_where = torch.where(above_cutoff)
    s = s[..., above_cutoff]
    v = v[..., above_cutoff]

    nps = snp[..., above_cutoff.detach().numpy()]
    vnp = vnp[..., above_cutoff.detach().numpy()]

    s_close = np.isclose(s.detach().numpy(), snp)
    v_close = np.isclose(v.detach().numpy(), vnp)
    # compose the square root matrix
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)

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
        x = x - x.mean(dim=1).unsqueeze(1)
        x = x / x.std(dim=1).unsqueeze(1)
        # cov = x.unsqueeze(2) @ x.unsqueeze(1)
        # cov_at_minus_half = symsqrt(cov)
        # x = cov_at_minus_half @ x
        x = torch.pow(x, 4) -3
        x = torch.abs(x)
        x = torch.sum(x,dim=1)
        return -x.mean()


if __name__ == '__main__':
    x = torch.Tensor(np.random.rand(5, 10))
    Criterion = KurtosisLoss()
    loss = Criterion(x)
