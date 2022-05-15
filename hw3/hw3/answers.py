r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=150, 
        seq_len=64,
        h_dim=256, 
        n_layers=3,
        dropout=0.2,
        learn_rate=0.001,
        lr_sched_factor=0.3,
        lr_sched_patience=3
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "The Chinese epidemic" #
    temperature = .1
    # ========================
    return start_seq, temperature


# Why do we split the corpus into sequences instead of training on the whole text?
part1_q1 = r"""
**Your answer:**
<font color='darkgreen'>

We split the corpus into sequences, so the learning will be limited for those sequences and that will help us to avoid overfitting. 
If we didn't do it, and use the whole text for training, the model will be memorizing parts of the input more then it is learning - that is overfitting in the context of text.

"""

# How is it possible that the generated text clearly shows memory longer than the sequence length?
part1_q2 = r"""
**Your answer:**
<font color='darkgreen'>

The model starts with an initial sequence and produces the prediction distribution of the next charcter, this output and the hidden state are used again in the next step of the model - this is used as a context for the next prediction.
Thus, the model keeps the context between time steps and then is able to generate text that is longer than the sequence length.

"""

# Why are we not shuffling the order of batches when training?
part1_q3 = r"""
**Your answer:**
<font color='darkgreen'>

The hidden state in the model keeps the context of each sequence and uses the order in which the sequences are evaluated.
The network's memory is built using the sequence and its order, hence, if we shuffled the batches, the model will "turn-off" the hidden states (context).
So, we won't shuffle the batches to keep the context between sequential batches, which improves the model's ability to generate words that are relevant to the context.

"""

# 1. Why do we lower the temperature for sampling (compared to the default of  1.0  when training)?
# 2. What happens when the temperature is very high and why?
# 3. What happens when the temperature is very low and why?
part1_q4 = r"""
**Your answer:**
<font color='darkgreen'>


Softmax with temperature:  
$$ \mathrm{softmax}_T(\vec{y}) = \frac{e^{\vec{y}/T}}{\sum_k e^{y_k/T}} $$  

Softmax with temperature is well known in the field of reinforcement learning, the temperature $T$ is used to tweak  between exploration and exploitation (mathematically, how close to uniform distributions are we):  

* Low temperatures $T$ - less uniform distributions, that put emphasis on actions with the highest probability (exploit), in order to accumulate experience. 
* High temperatures $T$ - more uniform distributions, that will let the model to choose actions that less tested before, to gain experience (explore).

1. When sampling in our model, we want the model to have higher chance to choose a char with highest score and reduce the sensetivity. Thus, we prefer lower temperature, to let the model applies step according to what learned (that also reduce a accumulated error).

2. If $T \longrightarrow \infty$, the model will pick the next char randomally (each char will have the same probability, explore regime).

3. If $T \longrightarrow 0$, the model will pick the next char with the largest expected probability to have greater score (higher spikes in the probability plot, exploit regime).

"""

# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = {
        'batch_size': 8,
        'h_dim': 32,
        'z_dim': 8,
        'x_sigma2': 0.002,
        'learn_rate': 0.0005,
        'betas': (0.9, 0.99)
    }
    # ========================
    return hypers


# What does the  σ2  hyperparameter (x_sigma2 in the code) do? Explain the effect of low and high values.
part2_q1 = r"""
**Your answer:**
<font color='darkgreen'>

The $\sigma^2$ hyperparameter controls the data-reconstruction loss, in other words, the randomness of the model. 
    
* Low values of $\sigma^2$ puts more weight on minimizing the data-reconstruction loss. Therefore, the model will try to construct outputs that similar to the dataset used in the training phase.  
    
* High values of $\sigma^2$ puts more weight on minimizing the Kullback-Liebler divergence $\mathcal{D}_{\mathrm{KL}}$ which means learning a distribution for the latent space and reduce overfitting to the training data. in other words, that make the encoded distributions closer to a standard normal distribution.


"""

# 1. Explain the purpose of both parts of the VAE loss term - reconstruction loss and KL divergence loss.
# 2. How is the latent-space distribution affected by the KL loss term?
# 3. What's the benefit of this?
part2_q2 = r"""
**Your answer:**
<font color='darkgreen'>

1. The VAE loss aims to minimize two elements - reconstruction term and a regularization term.
    * Minimizing the reconstruction loss element constrains the model in such a way, that it keeps the most information when encoding and decoding the data. which means that generated images will look real.
    * The Kullback-Liebler divergence loss element makes sure that the encoder output is a distribution that is close to a standard normal distribution - regularization term.

2. The Kullback-Liebler divergence loss regularizes how the encoder forms the latent space, and prefers a standard normal distribution for that. 

3. By minimizing this loss function, our model learns how to encode and the decode the data, and makes sure that the decoded instances in the latent space will have a distribution as chosen (in our case normal distribution), and also imitates the learned data.
In addition, another benefit of enforcing that distribution to be normal is that this will make the latent space easy for sampling - and neighboring points in the latent space will yield simillar contnent.

"""

# In the formulation of the VAE loss, why do we start by maximizing the evidence distribution,  𝑝(X) ?
part2_q3 = r"""
**Your answer:**
<font color='darkgreen'>

We are intrested in a subspapce of the instance space - the instance space is high demonentional and we are intrested only in the subspace which is simillar to our data.
We would like to generate a sample which resembles the subspace of the instance space which we have.
Therefore we maximize the evidence distribution $p(X)$.
In other words, we want our samples look like our training set, so we maximize the the distribution over the training set.



"""

# In the VAE encoder, why do we model the log of the latent-space variance corresponding to an input,  𝜎2𝛼 , instead of directly modelling this variance?
part2_q4 = r"""
**Your answer:**
<font color='darkgreen'>

The reason of using log instead of the value $\sigma$ for learning is the following:

1. Since $\sigma_\alpha^2$ is the variance it is a non negative quantity. by training $log(\sigma_\alpha^2)$ we ensure that sigma is a non negative value, because taking the exponent of a number is a positive quantity.
The exponent of $log(\sigma_\alpha^2)$ is $\sigma_\alpha^2$.

2. It is easier to train and more stable - since using $log$, maps number from the interval $[0,1]$ to $[log(1), -\infty]$, so instead of working with very small numbers in the interval $[0,1]$, we would work with big numbers in the interval $[log(1), -\infty]$.

"""

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
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


# Explain in detail why during training we sometimes need to maintain gradients when sampling from the GAN, and other times we don't. When are they maintained and why? When are they discarded and why?
part3_q1 = r"""
**Your answer:**
<font color='darkgreen'>

In the GAN training phase we aim to train both the generator and discriminator (one against the other).  
* When training the generator, we need to update its parameters (only its parameters) to improve the performance, so the gradients are maintained. 

* The generator output is then passed through the discriminator that tries to detect if this sample is real or fake. In order to train the discriminator, the output of the generator is used as is, and the generator's parameters are not needed to be updated so the gradients are discarded. 


"""

# 1. When training a GAN to generate images, should we decide to stop training solely based on the fact that the Generator loss is below some threshold? Why or why not?

# 2. What does it mean if the discriminator loss remains at a constant value while the generator loss decreases?
part3_q2 = r"""
**Your answer:**
<font color='darkgreen'>

1. Use a stopping criterion based only on generator's loss that reaches some threshold is not a good idea, because that can be caused by poor dicriminator, that can't point which data is fake. For example, a disriminator that thinks that every image is real will satisfy this criterion even if that generator output noise - so both the generator and the disctiminator will be poor.

2. In this case, we can understand that the generator works well and generates samples that look real by the disriminator. It can be because the discriminator works well and the generator output is really looking real, or if the generator is faster than the discriminator that doesn't work good enough.


"""

part3_q3 = r"""
**Your answer:**
<font color='darkgreen'>

The VAE's results are significantly more smooth but blurry, where the GAN's results are more noisy and sharp with more featues from the training set.

Those differences are caused by the fact that VAE tries to build a gaussian distribution that fits the data and the GAN is trying to learn a distribution of the data. That causes the VAE to be more smooth and blurry and the GAN to be more sharp and noisy with more details.


"""

# ==============
