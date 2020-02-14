from main import get_loaders, get_configuration
from AutoEncoder import autoencoder
import os
import torch
import pickle
import numpy as np


def analayze(experiment_name):
    batch_size, data_dir, device, dims, epochs, _, im_dim, image_format_ext, intial_lr, lamda, lr_decay, lr_decay_step_size, momentum, num_images_to_output, output_dir, print_every_iteration, write_output_every_epoch = get_configuration()
    net = autoencoder(dims)
    experiment_dir = os.path.join(output_dir, experiment_name)
    epochs = [int(x) for x in os.listdir(experiment_dir) if x.isnumeric()]
    max_epoch_dir = os.path.join(experiment_dir, str(max(epochs)))
    net.load_state_dict(torch.load(os.path.join(max_epoch_dir, 'weights.pth')))
    loaders = get_loaders(batch_size, data_dir)

    net.eval()

    mean_dict = {}
    for phase, loader in loaders.items():
        sum_means = np.zeros(dims[-1])
        for iteration, batch in enumerate(loader):
            batch_images, _ = batch
            batch_vectors = torch.flatten(batch_images, start_dim=1)
            latent_dim = net.encode(batch_vectors).detach().cpu().numpy()
            sum_means += np.mean(latent_dim, axis=0)
        mean_dict[phase] = sum_means / len(loader)

    mean_dump_path = os.path.join(experiment_dir, 'mean_dict.pkl')
    with open(mean_dump_path, 'wb') as f:
        pickle.dump(mean_dict, f)

    cov_dict = {}
    for phase, loader in loaders.items():
        sum_corr = np.zeros((dims[-1],dims[-1]))
        for iteration, batch in enumerate(loader):
            batch_images, _ = batch
            batch_vectors = torch.flatten(batch_images, start_dim=1)
            batch_size = batch_vectors.shape[0]
            latent_dim = net.encode(batch_vectors).detach().cpu().numpy() - mean_dict[phase]
            current_corr = np.multiply(np.expand_dims(latent_dim, -1), np.expand_dims(latent_dim, 1)) / batch_size
            sum_corr += np.mean(current_corr, axis=0)
            cov_dict[phase] = sum_corr / len(loader)

    correlation_dump_path = os.path.join(experiment_dir,'cov_dict.pkl')
    with open(correlation_dump_path,'wb') as f:
        pickle.dump(cov_dict,f)


if __name__ == '__main__':
    analayze('no_kurt_deeper_higher_lr')

