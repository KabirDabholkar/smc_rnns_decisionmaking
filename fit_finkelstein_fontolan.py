import numpy as np
import torch
import sys, os
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

from vi_rnn.datasets import Basic_dataset,Basic_dataset_with_trials
from utils import load_binned_trials_from_h5

datapath = "/Users/kabir/Documents/datasets/000060/sub-353936/sub-353936_ses-20170625_behavior+ecephys+ogen_binned_20ms.h5"

# load data
data_all, binned_spikes, metadata = load_binned_trials_from_h5(datapath)
data_all = data_all.astype(np.float32)
data_train, data_eval = train_test_split(data_all, test_size=0.2, random_state=seed)

task_params = {
    "name": "finkelstein_fontolan_sub-353936_ses-20170625",
    "dur": data_all.shape[-1],  # we will sample pseudo trials of "dur" timesteps during training
    "n_trials": data_train.shape[0],  # every epoch consists of 256 psuedo trials
}

dataset = Basic_dataset_with_trials(
    task_params=task_params,
    data=data_train,
    data_eval=data_eval,
    stim=None,  # you could additionally pass input / stimuli like this
    stim_eval=None,
)

# split into train and eval
# eval_split_ts = int(0.75 * data_all.shape[1])
# data_train = data_all[:, :eval_split_ts]
# data_eval = data_all[:, eval_split_ts:]


# # initialise a dataset class
# task_params = {
#     "name": "tutorial_cont",
#     "dur": 100,  # we will sample pseudo trials of "dur" timesteps during training
#     "n_trials": 400,  # every epoch consists of 256 psuedo trials
# }
# dataset = Basic_dataset(
#     task_params=task_params,
#     data=data_train,
#     data_eval=data_eval,
#     stim=None,  # you could additionally pass input / stimuli like this
#     stim_eval=None,
# )

# # plot some example data
fig, ax = plt.subplots(1, 1, figsize=(6, 2))
ax.imshow(dataset.__getitem__(0)[0], aspect="auto", cmap="Blues", vmax=2)
ax.set_xlabel("timesteps")
ax.set_ylabel("neurons")
fig.savefig("plots/spiking_example.png")

############ Initialize VAE ############

from vi_rnn.vae import VAE

enc_params = {
    "kernel_sizes": [21, 11, 1],  # kernel sizes of the CNN
    "padding_mode": "constant",  # padding mode of the CNN (e.g., "circular", "constant", "reflect")
    "nonlinearity": "gelu",  # "leaky_relu" or "gelu"
    "n_channels": [
        64, 
        64, 
    ],  # number of channels in the CNN (last one will be equal to dim_z)
    "init_scale": 0.1,  # initial scale of the noise predicted by the encoder
    "constant_var": False,  # whether or not to use a constant variance (as opposed to a data-dependent variance)
    "padding_location": "acausal",
}  # padding location of the CNN ("causal", "acausal", or "windowed")


rnn_params = {
    # transition and observation
    "transition": "low_rank",  # "low_rank" or "full_rank" RNN
    "observation": "one_to_one",  # "one_to_one" mapping between RNN and observed units or "affine" mapping from the latents
    # observation settings
    "readout_from": "currents",  # readout from the RNN activity before / after applying the non-linearty by setting this to "currents" / "rates" respectively.
    "train_obs_bias": True,  # whether or not to train a bias term in the observation model
    "train_obs_weights": True,  # whether or not train the weights of the observation model
    "obs_nonlinearity": "softplus",  # can be used to rectify the output (e.g., when using Poisson observations, use "softplus")
    "obs_likelihood": "Poisson",  # observation likelihood model ("Gauss" or "Poisson")
    # transition settings
    "activation": "relu",  # set the nonlinearity to "clipped_relu, "relu", "tanh" or "identity"
    "decay": 0.9,  # initial decay constant, scalar between 0 and 1
    "train_neuron_bias": True,  # train a bias term for every neuron
    "weight_dist": "uniform",  # weight distribution ("uniform" or "gauss")
    "initial_state": "trainable",  # initial state ("trainable", "zero", or "bias")
    "simulate_input": False,  # set to True when using time-varying inputs
    # noise covariances settings
    "train_noise_z": True,  # whether or not to train the transition noise scale
    "train_noise_z_t0": True,  # whether or not to train the initial state noise scale
    "init_noise_z": 0.1,  # initial scale of the transition noise
    "init_noise_z_t0": 0.1,  # initial scale of the initial state noise
    "noise_z": "diag",  # transition noise covariance type ("full", "diag" or "scalar"), set to "full" when using the optimal proposal
    "noise_z_t0": "diag",  # initial state noise covariance type ("full", "diag" or "scalar"), set to "full" when using the optimal proposal
}


VAE_params = {
    "dim_x": 51,  # observation dimension (number of units in the data)
    "dim_z": 2,  # latent dimension / rank of the RNN
    "dim_N": 51,  # amount of units in the RNN (can generally be different then the observation dim)
    "dim_u": 0,  # input stimulus dimension
    "enc_architecture": "CNN",  # encoder architecture (not trained when using linear Gauss observations)
    "enc_params": enc_params,  # encoder paramaters
    "rnn_params": rnn_params,  # parameters of the RNN
}

# initialise the VAE
vae = VAE(VAE_params)



############ Fit VAE ############

from vi_rnn.saving import save_model, load_model
from vi_rnn.train import train_VAE

training_params = {
    "lr": 1e-3,  # learning rate start
    "lr_end": 1e-5,  # learning rate end (with exponential decay)
    "n_epochs": 2,  # number of epochs to train
    "grad_norm": 0,  # gradient clipping above certain norm (if this is set to >0)
    "batch_size": 10,  # batch size
    "cuda": False,  # train on GPU
    "k": 64,  # number of particles to use
    "loss_f": "smc",  # use regular variational SMC ("smc"), or use the optimal ("opt_smc")
    "resample": "systematic",  # type of resampling "systematic", "multinomial" or "none"
    "run_eval": False,  # run an evaluation setup during training (requires additional parameters)
    "t_forward": 0,  # timesteps to predict without using the encoder
}


# run training

train_VAE(
    vae,
    training_params,
    dataset,
    sync_wandb=False,
    out_dir="output_data",
    fname=task_params["name"],
)