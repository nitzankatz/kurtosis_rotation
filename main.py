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
from tensorboardX import SummaryWriter


def main():
    batch_size, data_dir, device, dims, epochs, experiment_name, im_dim, image_format_ext, intial_lr, lamda, lr_decay, lr_decay_step_size, momentum, num_images_to_output, output_dir, print_every_iteration, write_output_every_epoch = get_configuration()

    net = autoencoder(dims)
    kurt_loss = KurtosisLoss()
    reconstruction_loss = torch.nn.MSELoss()

    loaders = get_loaders(batch_size, data_dir)

    writer = SummaryWriter(os.path.join(output_dir, experiment_name, 'tensorboard'))

    net.to(device)
    # optimizer = optim.SGD(net.parameters(), lr=intial_lr, momentum=momentum)
    optimizer = optim.Adam(net.parameters(), lr=intial_lr, betas=(0.0, 0.9), weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay)
    for epoch in range(epochs):
        loss_dict = {}
        for phase, loader in loaders.items():
            iter_in_epoch = len(loader)
            if phase == "train":
                net.train()
            else:
                net.eval()
            num_samples = 0
            loss_sum = 0
            for iteration, batch in enumerate(loader):
                if phase == 'train':
                    net.zero_grad()
                batch_images, _ = batch
                batch_vectors = torch.flatten(batch_images, start_dim=1)
                batch_vectors.to(device)
                latent_dim_variables = net.encode(batch_vectors)
                if phase =='train':
                    writer.add_histogram('latent',latent_dim_variables,epoch * iter_in_epoch + iteration)
                out = net.decode(latent_dim_variables)
                mse = reconstruction_loss(out, batch_vectors)
                kurt = lamda * kurt_loss(latent_dim_variables)
                loss = mse + kurt
                num_samples += 1
                loss_sum += loss.detach().cpu().numpy()
                if iteration % print_every_iteration == 0:
                    print(
                        'epoch {} ---- {} --- iteration {} out of {}. lr = {:.3f} mse = {:.4f} loss_kurt = {:.5f}'.format(epoch, phase,
                                                                                                       iteration,
                                                                                                       iter_in_epoch,
                                                                                                       optimizer.param_groups[
                                                                                                           0][
                                                                                                           'lr'],
                                                                                                       mse.detach().numpy(),
                                                                                                       kurt.detach().numpy()))
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            if epoch % write_output_every_epoch == 0:
                avg_loss = loss_sum / num_samples
                loss_dict[phase] = avg_loss
                epoch_dir = os.path.join(output_dir, experiment_name, str(epoch))
                cur_dir = os.path.join(epoch_dir, phase)
                os.makedirs(cur_dir, exist_ok=True)
                for i in range(num_images_to_output):
                    image = out[i, :].reshape(im_dim, im_dim)
                    orig_image = batch_vectors[i, :].reshape(im_dim, im_dim)
                    file_name = os.path.join(cur_dir, str(i) + image_format_ext)
                    orig_file_name = os.path.join(cur_dir, str(i) + '_orig' + image_format_ext)
                    save_image(image, file_name, normalize=True)
                    save_image(orig_image, orig_file_name, normalize=True)
                torch.save(net.state_dict(), os.path.join(epoch_dir, 'weights.pth'))
        writer.add_scalars("loss vs epoch", loss_dict, epoch)
        writer.add_scalar('lr vs epoch', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step()


def get_configuration():
    device = 'cpu'
    im_dim = 28
    dim = im_dim ** 2
    dims = [dim] * 6
    epochs = 100
    lamda = 0
    batch_size = 100
    print_every_iteration = 100
    write_output_every_epoch = 1
    num_images_to_output = 10
    intial_lr = 1e-3
    momentum = 0.9
    lr_decay_step_size = 10
    lr_decay = 0.5
    output_dir = 'output'
    data_dir = 'data'
    experiment_name = 'temp_adam_over_fit'
    image_format_ext = '.png'
    return batch_size, data_dir, device, dims, epochs, experiment_name, im_dim, image_format_ext, intial_lr, lamda, lr_decay, lr_decay_step_size, momentum, num_images_to_output, output_dir, print_every_iteration, write_output_every_epoch


def get_img_trans():
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5), (0.5))
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return img_transform


def get_loaders(batch_size, data_dir):
    img_transform = get_img_trans()
    dataset = MNIST(data_dir, transform=img_transform, train=True)
    testset = MNIST(data_dir, transform=img_transform, train=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    loaders = {'train': dataloader, 'test': testloader}
    return loaders


if __name__ == '__main__':
    main()
