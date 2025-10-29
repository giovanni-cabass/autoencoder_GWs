import sys
import os
import copy
import pickle
import time
###################################################
import numpy as np #type: ignore
import pandas as pd #type: ignore
import matplotlib.pyplot as plt #type: ignore
from tqdm.auto import tqdm #type: ignore
###################################################
import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.optim as optim #type: ignore
import torch.nn.functional as F #type: ignore
import torch.utils.data as utils_data #type: ignore
import torch.distributed as dist #type: ignore
from torch.distributions import constraints #type: ignore
from torch.distributions import MultivariateNormal #type: ignore
import torch.multiprocessing as mp #type: ignore
from torch.nn.parallel import DistributedDataParallel as DDP #type: ignore
###################################################
torch.set_default_dtype(torch.float32)
########################################################
import itertools #type: ignore

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

ntot=8192
n_train=4111
n_valid=4081
dims=3
latent_dims=2

##########################################################################################################################

def build_mlp(input_dim, hidden_dim, output_dim, layers, activation=nn.GELU()):
    """Create an MLP from the configuration."""
    seq = [nn.Linear(input_dim, hidden_dim), activation]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation]
    seq += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*seq)

##########################################################################################################################

class Autoencoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, latent_dim=2, layers=4):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = build_mlp(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=latent_dim, layers=layers)
        # Decoder
        self.decoder = build_mlp(input_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim, layers=layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

##########################################################################################################################

class CustomLoss_Jacobian_at_p(nn.Module):

    def __init__(self, alpha=1., beta=1.):
        super(CustomLoss_Jacobian_at_p, self).__init__()
        self.mse_loss = nn.MSELoss()  # Inherit MSE for reconstruction loss
        self.alpha = alpha  # Weight for reconstruction loss
        self.beta = beta  # Weight for matrix loss

    def forward(self,
                ####################
                y_pred,
                y_true,
                ####################
                decoder_fun,
                #encoder_fun,
                #point_p,
                #--> already have the encoded here, I impose at all points...
                latent_point,
                metric_target,
                npts
                ):
        # Reconstruction Loss
        reconstruction_loss = self.mse_loss(y_pred,y_true)

        #####################################################################################################################

        #latent_point = encoder_fun(point_p) #--> already have the encoded here, I impose at all points...
        #print(latent_point.requires_grad) #--> of course it gives True...
        ###########################################################
        #latent_point = latent_point.detach().requires_grad_(True)#
        ###########################################################
        jacobian_p = torch.autograd.functional.jacobian(decoder_fun, latent_point, create_graph=True)
        jacobian_p = jacobian_p[torch.arange(npts),:,torch.arange(npts),:]
        metric_p = torch.einsum("bij,bik->bjk", jacobian_p, jacobian_p)
        j_loss = torch.mean(torch.norm(metric_p-metric_target, p='fro',dim=(1,2)))

        # Combined Loss
        combined_loss = self.alpha * reconstruction_loss + self.beta * j_loss
        return combined_loss

##########################################################################################################################

def ddp_training_loop(rank, world_size, train_dataset, valid_dataset, n_epochs, configurable_batch_size, shared_dict):

    # Initialize lists to store losses for plotting
    training_losses = []
    validation_losses = []

    # Check if required environment variables for DDP are set
    if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
        raise RuntimeError("Environment variables MASTER_ADDR and MASTER_PORT must be set for DDP.")

    time.sleep(10)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )

    autoencoder_training = Autoencoder(input_dim=dims,hidden_dim=64,latent_dim=2,layers=4)
    autoencoder_training.to(rank)

    # =========================================================================
    # ADDED CHECK: Verify Model Dtype (Graceful Exit)
    # =========================================================================

    # Prepare a signal tensor (0 = OK to proceed, 1 = Error, everyone exit)
    exit_signal_tensor = torch.tensor(0, device=rank)
    # Note: Initializing on all ranks is essential for broadcast

    if rank == 0:
        print("\n--- Verifying Model Dtype ---")
        all_parameters_double = True

        print(f"Checking parameter data types for rank {rank}:")

        for name, param in autoencoder_training.named_parameters():
            # Check Dtype
            if param.dtype != torch.float32:
                print(f" X Parameter '{name}' is not float32 (is {param.dtype})")
                all_parameters_double = False
            else:
                print(f" V Parameter '{name}' is float32.")

        if not all_parameters_double:
            print("\n!!! WARNING: MODEL TYPE MISMATCH DETECTED. SIGNALING ALL RANKS TO HALT. !!!")
            # Set the signal to 1 (Error)
            exit_signal_tensor = torch.tensor(1, device=rank)
        else:
            print("\nSUCCESS: All parameters verified as float32. Proceeding with training.")

    # COLLECTIVE COMMUNICATION: Broadcast the signal from Rank 0 to all others
    # This implicitly acts as a barrier for all ranks.
    if dist.is_initialized():
        dist.broadcast(exit_signal_tensor, src=0)

    # Now, all ranks check the broadcasted signal
    if exit_signal_tensor.item() != 0:
        # Use 'raise' to jump straight to the 'finally' block for cleanup
        raise RuntimeError(f"Rank {rank}: Model verification failed on Rank 0. Exiting gracefully.")

    # If successful, the code proceeds normally
    # =========================================================================

    autoencoder_ddp=DDP(autoencoder_training,device_ids=[rank])
    params=autoencoder_ddp.parameters()

    criterion = CustomLoss_Jacobian_at_p().to(rank)

    ### OPTIMIZER CHOICES ###
    optimizer = optim.AdamW(params, lr=1.e-3,weight_decay=1.e-6)
    ### OPTIMIZER CHOICES ###

    ### SCHEDULER CHOICES ###
    #--> notice that ReduceLROnPlateau is called with "scheduler.step(avg_val_loss_global)", the other without anything ~>
    #--> see below!
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99995, patience=50)
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2)
    ### SCHEDULER CHOICES ###

    train_sampler = utils_data.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = utils_data.DataLoader(
        train_dataset,
        batch_size=configurable_batch_size,
        sampler=train_sampler
    )

    valid_sampler = utils_data.DistributedSampler(
        valid_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    valid_loader = utils_data.DataLoader(
        valid_dataset,
        batch_size=configurable_batch_size,
        sampler=valid_sampler
    )

    # Variables to track the best validation loss and the model's state
    #--> only this one needed...
    best_val_loss_to_save = float('inf')

    try:

        #for epoch in tqdm(range(n_epochs)):
        for epoch in tqdm(range(n_epochs), disable=(rank != 0)):
            # Set the epoch for the DistributedSampler to ensure correct shuffling
            train_sampler.set_epoch(epoch)
            #--> NOT NEEDED FOR VALIDATION SINCE SHUFFLE IS FALSE!!!

            # ------------------- Training Phase -------------------
            # Set models to training mode
            autoencoder_ddp.train()

            # A local variable to accumulate training loss for the epoch
            total_train_loss_local = 0.0

            for train_batch, metric_batch_train in train_loader:
                # Move data to the correct device for the current rank
                train_batch = train_batch.to(rank)
                metric_batch_train = metric_batch_train.to(rank)
                current_batch_size_train = train_batch.size(0)

                optimizer.zero_grad()

                # Calculate loss

                decoded_train, encoded_train = autoencoder_ddp(train_batch)

                loss = criterion(decoded_train,
                                 train_batch,
                                 autoencoder_ddp.module.decoder,
                                 encoded_train,
                                 metric_batch_train,
                                 current_batch_size_train)

                total_train_loss_local += loss.item()

                loss.backward()
                optimizer.step()

            # Calculate average training loss for the epoch on a per-process basis
            avg_train_loss_local = total_train_loss_local / len(train_loader)

            # Use all_reduce to get the average training loss across all GPUs
            # This tensor will hold the sum of the average losses from all ranks
            avg_train_loss_tensor = torch.tensor(avg_train_loss_local, device=rank, dtype=torch.float32)
            dist.all_reduce(avg_train_loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss_global = avg_train_loss_tensor.item() / world_size

            # Append the global training loss for this epoch
            training_losses.append(avg_train_loss_global)

            # ------------------- Validation Phase -------------------
            # Set models to evaluation mode
            autoencoder_ddp.eval()

            # A local variable to accumulate validation loss for the epoch
            total_val_loss_local = 0.0

            with torch.no_grad():
                for valid_batch, metric_batch_valid in valid_loader:
                    # Move data to the correct device for the current rank
                    valid_batch = valid_batch.to(rank)
                    metric_batch_valid = metric_batch_valid.to(rank)
                    current_batch_size_valid = valid_batch.size(0)

                    decoded_valid, encoded_valid = autoencoder_ddp(valid_batch)

                    # Calculate validation loss
                    val_loss = criterion(decoded_valid,
                                         valid_batch,
                                         autoencoder_ddp.module.decoder,
                                         encoded_valid,
                                         metric_batch_valid,
                                         current_batch_size_valid)
                    total_val_loss_local += val_loss.item()

            # Calculate average validation loss for the epoch on a per-process basis
            avg_val_loss_local = total_val_loss_local / len(valid_loader)

            # Use all_reduce to get the average validation loss across all GPUs
            # This tensor will hold the sum of the average losses from all ranks
            avg_val_loss_tensor = torch.tensor(avg_val_loss_local, device=rank, dtype=torch.float32)
            dist.all_reduce(avg_val_loss_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss_global = avg_val_loss_tensor.item() / world_size

            # ------------------- Scheduler Update -------------------
            # Only the master process needs to step the scheduler

            lr_tensor = torch.tensor(optimizer.param_groups[0]['lr'], device=rank, dtype=torch.float32)

            if rank == 0:

                old_lr = lr_tensor.item()

                # Rank 0 steps the scheduler, which changes its local optimizer LR
                scheduler.step(avg_val_loss_global)
                #--> first kind of scheduler...
                #scheduler.step()
                #--> cosine scheduler...

                new_lr_value = optimizer.param_groups[0]['lr']
                lr_tensor = torch.tensor(new_lr_value, device=rank, dtype=torch.float32)

                print(f"Epoch {epoch+1}/{n_epochs} | LR: {new_lr_value:.8f} (Change: {old_lr:.8f} -> {new_lr_value:.8f})")

            dist.broadcast(lr_tensor, src=0)

            new_lr = lr_tensor.item()

            optimizer.param_groups[0]['lr'] = new_lr

            # --> Add the barrier here for safety <--
            # if dist.is_initialized(): # A general good practice check
            #     dist.barrier()
            #--> unnecessary because dist.broadcast(lr_tensor, src=0) already gives a barrier!!!

            # Append the global validation loss for this epoch
            validation_losses.append(avg_val_loss_global)

            # ------------------- Logging and Saving -------------------
            # Only rank 0 handles logging and saving to avoid clutter and conflicts
            if rank == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss_global:.16f} | Validation Loss: {avg_val_loss_global:.16f}")
                # Save the best model
                if avg_val_loss_global < best_val_loss_to_save:
                    best_val_loss_to_save = avg_val_loss_global
                    print(f"New best validation loss: {best_val_loss_to_save:.16f}. Saving models...")
                    # Save the autoencoder model
                    torch.save(autoencoder_ddp.module.state_dict(),
                               os.path.join('cylinder_riemann.pth'))

    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        print(f"Rank {rank}: Received KeyboardInterrupt. Exiting...")
        pass

    finally:
        # Step 1: Check if the process group is initialized before doing anything.
        if dist.is_initialized():

            # Step 2: Only rank 0 saves the losses. This is a local operation.
            if rank == 0:
                shared_dict[0] = (training_losses, validation_losses)
                print(f"Losses saved to shared dictionary by Rank 0.")

            # Step 3: All ranks must destroy the process group.
            # This is a collective operation, so it must be called by all ranks.
            dist.destroy_process_group()
            print(f"Rank {rank}: Process group destroyed.")

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

if __name__ == "__main__":

    # Set environment variables
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '42429'
    os.environ['GLOO_SOCKET_IFNAME'] = 'lo'

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs found. DDP requires at least one GPU.")

    ################################################# TRAINING DATA PREPARATION #################################################

    train_data=torch.load(os.path.join('cylinder_train_data.pt'))
    train_metric=torch.eye(2).unsqueeze(0).repeat(n_train, 1, 1)

    ################################################# VALID DATA PREPARATION #################################################

    valid_data=torch.load(os.path.join('cylinder_valid_data.pt'))
    valid_metric=torch.eye(2).unsqueeze(0).repeat(n_valid, 1, 1)

    ################################################# MORE DATA PREPARATION #################################################

    # Create TensorDatasets from your prepared data
    train_dataset = utils_data.TensorDataset(train_data, train_metric)
    valid_dataset = utils_data.TensorDataset(valid_data, valid_metric)

    #n_epochs=20000
    n_epochs=10000
    configurable_batch_size=256#1000

    ################################################# START TRAINING #################################################

    print("Starting DDP training...")

    with mp.Manager() as manager:
        shared_dict = manager.dict()

        # Pass the shared dictionary to the spawn process
        p = mp.spawn(
            ddp_training_loop,
            args=(world_size, train_dataset, valid_dataset, n_epochs, configurable_batch_size, shared_dict),
            nprocs=world_size,
            join=True)

        print("DDP training finished. Plotting losses...")

        # Retrieve the losses from the shared dictionary
        if 0 in shared_dict:
            training_losses, validation_losses = shared_dict[0]

            print("Plotting losses...")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(training_losses, label='Training Loss')
            ax.plot(validation_losses, label='Validation Loss')
            ax.set_title('Loss Curves')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
            fig.savefig(os.path.join('cylinder_riemann.train_valid_losses.png'))
            print("Plot saved successfully.")
        else:
            print("No plot saved because no data was returned by rank 0.")
