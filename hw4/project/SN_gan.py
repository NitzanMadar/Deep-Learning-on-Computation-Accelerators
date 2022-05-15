import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
#         print(in_size)  # torch.Size([3, 64, 64]) --> C,H,W
        modules = []
    
        modules.append(nn.utils.spectral_norm(nn.Conv2d(in_size[0], 128, kernel_size = 5, stride = 2, padding = 2)))
#         modules.append(nn.BatchNorm2d(128))
        modules.append(nn.LeakyReLU(0.01))
        
        modules.append(nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size = 5, stride = 2, padding = 2)))
#         modules.append(nn.BatchNorm2d(256))
        modules.append(nn.LeakyReLU(0.01))
        
        modules.append(nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size = 5, stride = 2, padding = 2)))
#         modules.append(nn.BatchNorm2d(512))
        modules.append(nn.LeakyReLU(0.01))
        
        modules.append(nn.utils.spectral_norm(nn.Conv2d(512, 1024, kernel_size = 5, stride = 2, padding = 2)))
#         modules.append(nn.BatchNorm2d(1024))
        modules.append(nn.LeakyReLU(0.01))
                
        self.Discriminator_net = nn.Sequential(*modules)
        self.linear = nn.Linear(in_size[1] * in_size[2] * 4, 1)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        # discriminator and than linear
        y = self.linear(self.Discriminator_net(x).view(x.shape[0], -1))
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        modules = []
        self.first_out = 512
        self.z_dim = z_dim
#         print(z_dim) #128
#         print(out_channels)
#         modules.append(nn.Linear(z_dim,2**14))
        self.featuremap_size = featuremap_size
        modules.append(nn.ConvTranspose2d(1024, self.first_out, kernel_size = 5, stride = 2, padding = 2, output_padding = 1))
        modules.append(nn.BatchNorm2d(self.first_out))
        modules.append(nn.LeakyReLU(0.01))
        
        modules.append(nn.ConvTranspose2d(self.first_out, 256, kernel_size = 5, stride = 2, padding = 2, output_padding = 1))
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.LeakyReLU(0.01))
        
        modules.append(nn.ConvTranspose2d(256, 128, kernel_size = 5, stride = 2, padding = 2, output_padding = 1))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.LeakyReLU(0.01))
        
        modules.append(nn.ConvTranspose2d(128, out_channels, kernel_size = 5, stride = 2, padding = 2, output_padding = 1))

        modules.append(nn.Tanh())
        
        
        self.unlinear = nn.Linear(z_dim, featuremap_size ** 2 * 2**10)
        self.Decoder = nn.Sequential(*modules)
        self.shape = (2**10, featuremap_size, featuremap_size)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        torch.set_grad_enabled(with_grad)
        samples = self(torch.randn(n, self.z_dim, device=device))
        torch.set_grad_enabled(True)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        # [128, 256, 5, 5], but got 2-dimensional input of size [1, 128] instead
        # unlinear and than Decoder
        x=self.Decoder(self.unlinear(z).reshape((-1, *self.shape)))
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    device = y_data.device
    loss_fn = nn.BCEWithLogitsLoss()
    noise = label_noise / 2

    # "fuzzy"            rand [0, 1]*N - N/2
    y_data_target =      torch.rand_like(y_data) * label_noise - noise      + data_label
    y_generated_target = torch.rand_like(y_generated) * label_noise - noise + (1 - data_label)

    # loss
    loss_data =      loss_fn(y_data, y_data_target).to(device)
    loss_generated = loss_fn(y_generated, y_generated_target).to(device)
    
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    device = y_generated.device
    y_target = data_label*torch.ones_like(y_generated)
    loss = nn.BCEWithLogitsLoss()(y_generated, y_target)
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: DataLoader,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()
    x_gen = gen_model.sample(len(x_data), with_grad=True) # with grad or not???????
    dsc_gen_pred = dsc_model(x_gen.detach()) # try detach...
    dsc_data_pred = dsc_model(x_data)
    dsc_loss = dsc_loss_fn(dsc_data_pred, dsc_gen_pred)
    dsc_loss.backward()
    dsc_optimizer.step()
    
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()
    dsc_gen_pred = dsc_model(x_gen)
    gen_loss = gen_loss_fn(dsc_gen_pred)
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    if len(gen_losses) >= 2:# and gen_losses[-1] < gen_losses[-2]: 
        torch.save(gen_model,checkpoint_file)
        saved = True
    # ========================
    return saved
