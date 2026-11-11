import os, sys
import numpy as np # type: ignore
from numpy import amin,amax # type: ignore
import matplotlib as mpl # type: ignore
import matplotlib.pyplot as plt # type: ignore
import scipy, copy # type: ignore
from scipy import interpolate # type: ignore
from importlib import reload
import json, glob, time
import torch.multiprocessing as mp # type: ignore
from tqdm.auto import tqdm # type: ignore

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import matplotlib.pyplot as plt # type: ignore
from mpl_toolkits.mplot3d import Axes3D # type: ignore

from torch.utils.data import DataLoader # type: ignore
from torch.utils.data import DataLoader # type: ignore
import torch.nn.functional as F # type: ignore

from scipy.spatial.distance import pdist, squareform # type: ignore

from scipy.spatial import distance_matrix # type: ignore

#################################################################

# Define upper tri masking
def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask], mask

# Define the MLP architecture
def build_mlp(input_dim, hidden_dim, output_dim, layers, activation=nn.GELU()):
    """Create an MLP from the configuration."""
    seq = [nn.Linear(input_dim, hidden_dim), activation]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation]
    seq += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*seq)

# Define the Autoencoder model
class Autoencoder_suppress(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, latent_dim=2, layers=4):
        super(Autoencoder_suppress, self).__init__()
        # Encoder
        self.encoder = build_mlp(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=latent_dim, layers=layers)
        # Decoder
        self.decoder = build_mlp(input_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim, layers=layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class CustomLoss_suppress(nn.Module):
    def __init__(self, alpha=1., beta=1.):
        super(CustomLoss_suppress, self).__init__()
        self.mse_loss = nn.MSELoss()  # Inherit MSE for reconstruction loss
        self.alpha = alpha  # Weight for reconstruction loss
        self.beta = beta  # Weight for sparsity loss

    def forward(self, y_pred, y_true, embeddings, original_distances, mask_uptri):
        # Reconstruction Loss
        reconstruction_loss = self.mse_loss(y_pred,y_true)

        # Distance-Preserving Loss
        pairwise_distances = torch.norm(embeddings[:, None] - embeddings, dim=2)
        pairwise_distances = pairwise_distances[mask_uptri]

        #####################################################################################################################

        weights = torch.exp(-1.0*original_distances)
        distance_loss = self.mse_loss(weights*pairwise_distances, weights*original_distances)

        # Combined Loss
        combined_loss = self.alpha * reconstruction_loss + self.beta * distance_loss
        return combined_loss

def train_autoencoder(device_id,
                      train_data, original_distances_train, mask_uptri_train,
                      valid_data, original_distances_valid, mask_uptri_valid,
                      queue, dims=10, epochs=50000, lr=1e-3):

    # Set the device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    save_name=str(device_id)

    # Instantiate the model, loss, and optimizer
    autoencoder = Autoencoder_suppress(input_dim=dims, hidden_dim=64, latent_dim=2, layers=4).to(device)
    criterion = CustomLoss_suppress().to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

    # Move data to the device
    train_data_gpu = train_data.to(device)
    original_distances_train_gpu = original_distances_train.to(device)
    mask_uptri_train_gpu = mask_uptri_train.to(device)
    valid_data_gpu = valid_data.to(device)
    original_distances_valid_gpu = original_distances_valid.to(device)
    mask_uptri_valid_gpu = mask_uptri_valid.to(device)

    # Training & validation loop
    best_val_loss = float('inf')
    best_model = None
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = autoencoder(train_data_gpu)
        loss = criterion(output[0], train_data_gpu, output[1], original_distances_train_gpu, mask_uptri_train_gpu)
        loss.backward()
        optimizer.step()

        # Validation
        with torch.no_grad():
            val_help = autoencoder(valid_data_gpu)
            val_loss = criterion(val_help[0], valid_data_gpu, val_help[1], original_distances_valid_gpu, mask_uptri_valid_gpu)

            # Send progress updates to the queue
            queue.put((device_id, epoch + 1, loss.item(), val_loss.item()))

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(autoencoder.state_dict())

    # Save the best model to disk
    torch.save(best_model, './' + f'autoencoder_b_1_sb_'+save_name+'.pth')
    print(f"Training completed on device {device_id}. Best loss: {best_val_loss:.8f}")

#################################################################

if __name__ == "__main__":
    print(f"Script is being executed in process {os.getpid()}")
    print("This will only run in the main process!")
    mp.set_start_method('spawn', force=True)
    queue = mp.Queue()
    processes = []

    # Import data
    print("-> importing data, hopefully only once!")
    folder ="./"

    ca_SubBanks=[]
    phases_SubBanks=[]
    SVD_SubBanks=[]
    nbanks=1
    subbanks=[3]

    weights_banks=[]
    ############################################################
    bank_name=1
    for ib in range(nbanks):
        print("$$$$$$$$$$$$$$$")
        print(bank_name)
        print("###############")
        wts=np.loadtxt(folder+"weights_"+str(bank_name)+".txt")
        weights_banks.append(wts)
        _=[]
        __=[]
        ___=[]
        sub_bank_name=0
        #for isb in range(len(bins)):
        for isb in range(subbanks[ib]):
            print(sub_bank_name)
            name=str(bank_name)
            subname = str(sub_bank_name)
            ca=np.loadtxt(folder+"ca_bank_"+name+"_subbank_"+subname+".txt")
            phases=np.loadtxt(folder+"phases_bank_"+name+"_subbank_"+subname+".txt")
            Vph=np.loadtxt(folder+"Vph_bank_"+name+"_subbank_"+subname+".txt")
            mean_phase=np.loadtxt(folder+"mean_phase_bank_"+name+"_subbank_"+subname+".txt")
            _.append(ca)
            __.append(phases)
            ___.append([Vph, Vph, Vph, mean_phase])
            #--> Vph, Vph, Vph because before I was importing also Up, sp, Vph and I didn't want to look for everywhere I had to change the code...
            sub_bank_name = sub_bank_name+1
        ca_SubBanks.append(_)
        phases_SubBanks.append(__)
        SVD_SubBanks.append(___)
        bank_name = bank_name+1

    for i in range(3):
        print(ca_SubBanks[0][i].shape)
        print(phases_SubBanks[0][i].shape)
        print(SVD_SubBanks[0][i][2].shape)
        print(SVD_SubBanks[0][i][3].shape)
        print("#################################")

    ca_phases_Vph_mean_phase=[]
    for i in range(3):
        ca_phases_Vph_mean_phase.append([ca_SubBanks[0][i],
                                        phases_SubBanks[0][i],
                                        SVD_SubBanks[0][i][2],
                                        SVD_SubBanks[0][i][3]])

    weights=weights_banks[0]
    #--> this depends only on bank, not subbank...
    wt=weights#**0.5

    #--> recall that ind selects validation, ~ind selects training...
    #train_data = data[~ind]
    #valid_data = data[ind]

    dims = 10
    data=[]
    norm_factor=[]
    train_data=[]
    valid_data=[]
    ind=[]

    sub_bank_name=0
    for i in range(3):
        ca=ca_phases_Vph_mean_phase[i][0]
        ca_ = torch.tensor(ca[:, :dims], dtype=torch.float32)
        norm_factor_ = torch.std(ca_[:,0])
        np.savetxt('./'+"norm_factor_b_1_sb_"+str(sub_bank_name)+".txt", np.array([norm_factor_.numpy()]))
        ca_ = ca_/norm_factor_
        data.append(ca_)
        norm_factor.append(norm_factor_)
        np.random.seed(1648)
        ind_=np.random.choice([0, 1], size=(len(ca),), p=[1./2, 1./2]).astype('bool')
        train_data.append(ca_[~ind_])
        valid_data.append(ca_[ind_])
        ind.append(ind_)
        sub_bank_name = sub_bank_name+1

    print(data[0].shape)
    print(data[1].shape)
    print(data[2].shape)

    distances_train=[]
    distances_valid=[]
    masks_train=[]
    masks_valid=[]
    for i in range(3):
        orig_distances_train_temp          = squareform(pdist(train_data[i]        , metric="euclidean"))
        orig_distances_valid_temp          = squareform(pdist(valid_data[i]        , metric="euclidean"))
        orig_distances_train, mask_uptri_train = upper_tri_masking(orig_distances_train_temp)
        orig_distances_valid, mask_uptri_valid = upper_tri_masking(orig_distances_valid_temp)
        original_distances_train      = torch.tensor(orig_distances_train, dtype=torch.float32)
        original_distances_valid      = torch.tensor(orig_distances_valid, dtype=torch.float32)
        distances_train.append(original_distances_train)
        distances_valid.append(original_distances_valid)
        masks_train.append(torch.tensor(mask_uptri_train))
        masks_valid.append(torch.tensor(mask_uptri_valid))

    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Number of available GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. No GPUs detected or properly configured for PyTorch.")
    ######################################################################################################
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs: {', '.join([torch.cuda.get_device_name(i) for i in range(num_gpus)])}")

    # Launch training on each GPU in parallel
    mp.set_start_method('spawn', force=True)
    print("This will only run in the main process!")
    queue = mp.Queue()
    processes = []

    for i in range(3):
        print(f"Training started on device: {i}")
        p = mp.Process(target=train_autoencoder, args=(i,
                                                       train_data[i], distances_train[i], masks_train[i],
                                                       valid_data[i], distances_valid[i], masks_valid[i],
                                                       queue))
        p.start()
        processes.append(p)

    print("\n")
    print("\n")
    print("\n")

    # Monitor progress
    progress_bars = {i: tqdm(total=50000, desc=f"GPU {i}", position=i) for i in range(3)}
    while any(p.is_alive() for p in processes):
        while not queue.empty():
            message = queue.get()
            #print(f"Received from queue: {message}")
            if isinstance(message, tuple) and len(message) == 4:
                device_id, epoch, tloss, vloss = message
                # Update the progress bar for the corresponding GPU
                progress_bars[device_id].update(1)
                if epoch % 1000 == 0:
                    print(f'Device {device_id}, Epoch {epoch+1}, train loss: {tloss:.8f}, val loss: {vloss:.8f}')
            else:
                print(f"Queue message: {message}")

    for p in processes:
        p.join()

    # Close progress bars
    for bar in progress_bars.values():
        bar.close()
