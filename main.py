from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import torch
import torch.optim as optim
from AutoEncoder import autoencoder
from kurstosis_loss import KurtosisLoss
import os
from PIL import Image
import numpy

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
    im_dim = 28
    dim = im_dim ** 2
    dims = [dim] * 3
    epochs = 100
    lamda = 0
    batch_size = 100
    print_every_iteration = 100
    write_output_every_epoch = 1
    num_images_to_output = 10
    intial_lr = 0.01
    momentum = 0.9
    output_dir = 'output'
    train_output_dir_name = 'train'
    test_output_dir_name = 'test'
    data_dir = 'data'
    experiment_name = 'no_bottleneck_no_kurt_momentum'
    image_format_ext = '.png'
    train_dir = os.path.join(output_dir, experiment_name, train_output_dir_name)
    test_dir = os.path.join(output_dir, experiment_name, test_output_dir_name)

    net = autoencoder(dims)
    kurt_loss = KurtosisLoss()
    reconstruction_loss = torch.nn.MSELoss()

    dataset = MNIST(data_dir, transform=img_transform, train=True)
    testset = MNIST(data_dir, transform=img_transform, train=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=num_images_to_output, shuffle=True)

    iter_in_epoch = len(dataloader)

    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=intial_lr,momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    # batch = next(iter(dataloader))
    # for epoch in range(100):
    #     for iteration in range(6000):
    for epoch in range(epochs):
        for iteration, batch in enumerate(dataloader):
            net.zero_grad()
            batch_images, _ = batch
            batch_vectors = torch.flatten(batch_images, start_dim=1)
            batch_vectors.to(device)
            latent_dim_variables = net.encode(batch_vectors)
            out = net.decode(latent_dim_variables)
            loss = reconstruction_loss(out, batch_vectors) + lamda * kurt_loss(latent_dim_variables)
            if iteration % print_every_iteration == 0:
                print(
                    'epoch {} --- iteration {} out of {}. lr = {:.3f} loss = {}'.format(epoch, iteration, iter_in_epoch,
                                                                                        optimizer.param_groups[0][
                                                                                            'lr'],
                                                                                        loss.detach().numpy()))
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % write_output_every_epoch == 0:
            cur_epoch_train_dir = os.path.join(train_dir, str(epoch))
            cur_epoch_test_dir = os.path.join(test_dir, str(epoch))
            os.makedirs(cur_epoch_train_dir, exist_ok=True)
            os.makedirs(cur_epoch_test_dir, exist_ok=True)
            for i in range(num_images_to_output):
                image = out[i, :].reshape(im_dim, im_dim)
                orig_image = batch_vectors[i, :].reshape(im_dim, im_dim)
                file_name = os.path.join(cur_epoch_train_dir, str(i) + image_format_ext)
                orig_file_name = os.path.join(cur_epoch_train_dir, str(i) + '_orig' + image_format_ext)
                save_image(image, file_name, normalize=True)
                save_image(orig_image, orig_file_name, normalize=True)
                # test_images, _ = next(iter(testloader))

a = 3
