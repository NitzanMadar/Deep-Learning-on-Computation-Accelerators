
# ==============
# Vanilla GAN

def vanilla_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=32, 
        z_dim=128,
        data_label=1, 
        label_noise=0.25,
        discriminator_optimizer={'type': 'Adam', 'weight_decay': 0.02, 'betas': (0.5, 0.99), 'lr': 0.0002},
        generator_optimizer={'type': 'Adam', 'weight_decay': 0.02, 'betas': (0.5, 0.99), 'lr': 0.0002}
    )
    # ========================
    return hypers

# ==============

# ==============
# Spectral normalization GAN (SN-GAN)

def sn_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=32, 
        z_dim=128,
        data_label=1, 
        label_noise=0.25,
        discriminator_optimizer={'type': 'Adam', 'weight_decay': 0.02, 'betas': (0.5, 0.99), 'lr': 0.0002},
        generator_optimizer={'type': 'Adam', 'weight_decay': 0.02, 'betas': (0.5, 0.99), 'lr': 0.0002}
    )
    # ========================
    return hypers

# ==============

# ==============
# WGAN (using Wasserstein Loss)

def wgan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=32, 
        z_dim=10,
        data_label=1, 
        label_noise=0.25,
#         discriminator_optimizer={'type': 'Adam', 'weight_decay': 0.02, 'betas': (0.5, 0.99), 'lr': 0.0002},
#         generator_optimizer={'type': 'Adam', 'weight_decay': 0.02, 'betas': (0.5, 0.99), 'lr': 0.0002}
        
        discriminator_optimizer={'type': 'RMSprop', 
#                                  'weight_decay': 0.02,  
                                 'lr': 0.00005},
        generator_optimizer={'type': 'RMSprop', 
#                              'weight_decay': 0.02,  
                             'lr': 0.00005}
    )
    # ========================
    return hypers

# ==============

# ==============
# SN-WGAN (using Wasserstein Loss)

def sn_wgan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=32, 
        z_dim=5,
        data_label=1, 
        label_noise=0.25,
#         discriminator_optimizer={'type': 'Adam', 'weight_decay': 0.02, 'betas': (0.5, 0.99), 'lr': 0.0002},
#         generator_optimizer={'type': 'Adam', 'weight_decay': 0.02, 'betas': (0.5, 0.99), 'lr': 0.0002}
        
        discriminator_optimizer={'type': 'RMSprop', 
#                                  'weight_decay': 0.02,  
                                 'lr': 0.00005},
        generator_optimizer={'type': 'RMSprop', 
#                              'weight_decay': 0.02,  
                             'lr': 0.00005}
    )
    # ========================
    return hypers

# ==============