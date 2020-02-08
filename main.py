from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import torch
import torch.optim as optim
from AutoEncoder import autoencoder
from kurstosis_loss import KurtosisLoss

# if not os.path.exists('./mlp_img'):
#     os.mkdir('./mlp_img')
#
#
# def to_img(x):
#     x = 0.5 * (x + 1)
#     x = x.clamp(0, 1)
#     x = x.view(x.size(0), 1, 28, 28)
#     return x
#
#
# num_epochs = 100
# batch_size = 128
# learning_rate = 1e-3


if __name__ == '__main__':
    device = 'cpu'
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dim = 28 ** 2
    dims = [dim] * 3
    epochs = 3
    gamma = 0
    print_every = 100

    net = autoencoder(dims)
    kurt_loss = KurtosisLoss()
    reconstruction_loss = torch.nn.MSELoss()

    dataset = MNIST('data', transform=img_transform, train=True)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    iter_in_epoch = len(dataloader)


    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1200, gamma=0.5)
    batch = next(iter(dataloader))
    for epoch in range(100):
        for iteration in range(6000):
    # for epoch in range(epochs):
        # for iteration, batch in enumerate(dataloader):
            batch_images, _ = batch
            batch_vectors = torch.flatten(batch_images, start_dim=1)
            batch_vectors.to(device)
            latent_dim_variables = net.encode(batch_vectors)
            out = net.decode(latent_dim_variables)
            loss = reconstruction_loss(out, batch_vectors) + gamma * kurt_loss(latent_dim_variables)
            if iteration % print_every == 0:
                print('epoch {} --- iteration {} out of {}. lr = {} loss = {}'.format(epoch,iteration,iter_in_epoch,str(optimizer.param_groups[0]['lr']),str(loss.detach().numpy())))
            loss.backward()
            optimizer.step()
            scheduler.step()
a = 3
