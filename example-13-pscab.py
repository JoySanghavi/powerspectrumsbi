import os
import zarr
import ray
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import json5
import falcon
from falcon.contrib.norms import LazyOnlineNorm
import corner
import functions
import numpy as np
import tools21cm as t2c
torch.manual_seed(42)
np.random.seed(42)

# prior for A, n, gains
maxr=3
minr=-maxr

# simulator parameters

nu = 1420.4 #in Mhz
redshift = 9
numtel = 64
stdev = 0.5
mysize=128
gaussianimage = gaussian_2d(size=mysize, sigma=5)
ant_config = torch.load('/home/jsanghavi1/falcon/telecopelocation/paddedAA2_4_10_9_128.pth')
subset12 = functions.split_indices(int(numtel*(numtel-1)/2),2)
# time-named directory
current_time = datetime.now().strftime("%y%m%d_%H%M%S")
os.makedirs(current_time, exist_ok=True)
with open("shared_timestamp.txt", "w") as f:
    f.write(current_time)

# theta_obs is the correct parameter to be inferred on
# Create theta_obs with interleaved real/imag parts (shape: [batch_size, 2*numtel-1])
theta_obs = torch.normal(0, stdev, size=(1, 2 * (numtel-1)))
an_obs = torch.tensor([[1.0, -2.0]])
x_obs = functions.sampler13(theta_obs, an_obs, (128,128), gaussianimage, ant_config, subset12, noise=0, nu=nu, z=redshift)
B, C, H, W = x_obs.shape
# Assume mask shape is [C, H, W] (identical for all batches)
mask = (x_obs[0].real != 0) | (x_obs[0].imag != 0)  # or given mask
# Flatten mask and get indices of non-zero positions (same for all batches)
flat_mask = mask.reshape(-1)  # shape: [C*H*W]
indices = flat_mask.nonzero(as_tuple=False).squeeze()  # shape: [n]
# Flatten x_obs per batch: shape [B, C*H*W]
x_obs = x_obs.reshape(B, -1)
# Select same indices for every batch using advanced indexing
x_obs = x_obs[:, indices]  # shape: [B, n]

torch.save({"theta_obs": theta_obs, "x_obs": x_obs, "numtel": numtel},'/home/jsanghavi1/falcon/'+current_time+'/observations.pth')
# Start time
start_time = datetime.datetime.now().timestamp()

class gain:
    def sample(self, batch_dim, parent_conditions=[]):
        z = parent_conditions
        tensor1 = z.clone().detach().float()
        assert tensor1.shape[1] == 2*(numtel-1), "Input must have shape (batch_size, 12)"
        return tensor1

    def get_shape_and_dtype(self):
        return (2*(numtel-1),), 'float32'

class AN:
    def sample(self, batch_dim, parent_conditions=[]):
        noises = []
        for _ in range(batch_dim):
            noise = get_noise()
            noises.append(noise)
        return torch.stack(noises)

    def get_shape_and_dtype(self):
        return (2,), 'float32'

class gainandantouv:
    def sample(self, batch_dim, parent_conditions=[]):
        gain, an = parent_conditions
        x = functions.sampler13(gain, an, (mysize,mysize), gaussianimage, ant_config, subset12, noise=0, nu=nu, z=redshift)
        B, C, H, W = x.shape
        # Assume mask shape is [C, H, W] (identical for all batches)
        mask = (x[0].real != 0) | (x[0].imag != 0)  # or given mask
        # Flatten mask and get indices of non-zero positions (same for all batches)
        flat_mask = mask.reshape(-1)  # shape: [C*H*W]
        indices = flat_mask.nonzero(as_tuple=False).squeeze()  # shape: [n]
        # Flatten x_obs per batch: shape [B, C*H*W]
        x = x.reshape(B, -1)
        # Select same indices for every batch using advanced indexing
        x = x[:, indices]  # shape: [B, n]
        return x

    def get_shape_and_dtype(self):
        return (x_obs.shape[1],), 'float32'



class Simulate:
    def __init__(self):
        pass

    def sample(self, num_samples, parent_conditions=[]):
        z = parent_conditions[0]
        tensor1 = z.clone().detach().float()


        return x

    def get_shape_and_dtype(self):
        return (x_obs.shape[1],), 'float64'


class E(torch.nn.Module):
    def __init__(self):
        super(E, self).__init__()
        self.norm = LazyOnlineNorm(momentum=5e-3)
        #self.linear = torch.nn.LazyLinear(NPAR*2)

    def forward(self, x):
        x = self.norm(x).float()
        return x


#async def main():
def main():
    ### User defined code

    num_epochs = 40
    n_train = 4096*2
    gammapar=0.5
    discardbool=True
    early_stop_patiencepar = 15
    num_resimspar=512*2
    filepath = '/home/jsanghavi1/falcon/'+current_time+'/zarr.zarr'
    # Graph definition

    observations = {
        #"x": torch.as_tensor([[0.5, ]]),
        "x": x_obs,
    }

    # p(z) and q(z|x)
    priors = 2*(numtel+1)*[['uniform', minr, maxr]]#numtel*[('normal', 1, 2.5), ('normal', 0, 2.5)]


    node_z = falcon.Node("z",
                falcon.LazyLoader("falcon.contrib.NSF_tempering_gaussian.NSFNode", embeddings=[E]),
                parents=[], evidence=['x','an'],
                module_config=dict(priors = priors, device='cuda', num_epochs = num_epochs, discard_samples=discardbool, early_stop_patience=early_stop_patiencepar, gamma = gammapar,
                    lr_decay_factor=0.1, lr=1e-2),   # 0.1, 0.5, 1.0
                actor_config=dict(num_gpus=1)
                )  

    # p(x|z)       
    node_an = falcon.Node('an',
                falcon.LazyLoader("falcon.contrib.NSF_tempering_gaussian.NSFNode", embeddings=[E]),
                parents=[], evidence=['x'],
                module_config=dict(priors = priors, device='cuda', num_epochs = num_epochs, discard_samples=discardbool, early_stop_patience=early_stop_patiencepar, gamma = gammapar,
                    lr_decay_factor=0.1, lr=1e-2), 
                actor_config=dict(num_gpus=1)
                )

    node_x = falcon.Node("x",
                gainandantouv,
                parents=['z', 'an'],
                observed=True, resample=True)


    graph = falcon.Graph([node_z, node_an, node_x])
    print(graph)


    #########################
    ### Boiler plate code ###
    #########################

    # 0) Deploy graph
    print("Graph deploying")
    deployed_graph = falcon.DeployedGraph(graph)
    print('Graph deployed')
    # 1) Prepare dataset manager for deployed graph and store initial samples
    shapes_and_dtypes = deployed_graph.get_shapes_and_dtypes()
    print('Graph shapes done')
    dataset_manager = falcon.get_zarr_dataset_manager(shapes_and_dtypes, filepath,
            num_min_sims = n_train, num_val_sims=128, num_resims = num_resimspar)
    #time.sleep(1)
    print('Graph dataset made')
    ray.get(dataset_manager.generate_samples.remote(deployed_graph, num_sims = 1024))
    print("Network training")
    try:
        deployed_graph.train(dataset_manager, observations)
    except KeyboardInterrupt:
        pass

    print("Network trained")
    # 4) Evaluation and storage (here sample from the trained graph)
    samples = deployed_graph.conditioned_sample(5000, observations)
    torch.save(samples, '/home/jsanghavi1/falcon/'+current_time+'/samples.pth')
    plot_samples = samples['z']
    plot_samples = plot_samples.numpy()
    #plot_samples = torch.stack([samples['z'][:,0],  samples['z'][:,1]]).T
    # labels = interleave_ri(numtel)
    # fig = corner.corner(plot_samples, labels=labels, bins=50, smooth=1.0, levels=[0.6827, 0.9545, 0.9973],
    #                 plot_contours=True, color='royalblue',  
    #                 fill_contours=False, show_titles=True, title_fmt=".2f")
    # plt.gca().tick_params(axis='both', labelsize=10)
    # plt.gca().title.set_fontsize(12)
    # corner.overplot_lines(fig, theta_obs.numpy()[0], color="C1")
    # corner.overplot_points(fig, theta_obs.numpy()[0][None], marker="s", color="C1")
    # #pairplot(plot_samples, limits=[[-5*SIGMA, 5*SIGMA]]*NPAR, figsize=(10, 10))
    # plt.savefig('/home/jsanghavi1/falcon/'+current_time+'/figures.png', dpi=500, bbox_inches="tight")
    # plt.show()

    fig, axes = plt.subplots(numtel, 1, figsize=(6, 2*numtel), sharex=True)
    # standard_normal_samples = torch.randn(10000).numpy()
    for i in range(numtel):
        ax = axes[i]
        sns.kdeplot(plot_samples[:, i], ax=ax, fill=True, color="blue", label=f'Parameter {i+1}')
        # sns.kdeplot(standard_normal_samples, ax=ax, color="black", linestyle='--', label='Standard Normal')
        # ax.set_title(f'pdf {i+1} 50/100 sig, gam 0.5, 500/100 epochs, discard')
        ax.legend()

    plt.xlabel("Parameter Value")
    plt.tight_layout()
    plt.savefig('/home/jsanghavi1/falcon/'+current_time+'/figures.png')
    plt.show()

    # 5) Clean up Ray resources
    deployed_graph.shutdown()
    root = zarr.open_group(filepath, mode='r')
    # Get the length (number of samples) from one dataset, e.g., 'z'
    num_entries = root['z'].shape[0]
    end_time = datetime.datetime.now().timestamp()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    # Example data
    # data = {
    #     'time': elapsed_time,
    #     'parameters': numtel,
    #     'sigma': stdev,
    #     'epoch': num_epochs,
    #     'initdata': n_train,
    #     'noise': noise,
    #     'loopdata': num_resimspar,
    #     'gamma': gammapar,
    #     'discard': discardbool,
    #     'early_stop': early_stop_patiencepar,
    #     'totdata': num_entries
    # }
    # # Save data to a JSON5 file
    # with open('/home/jsanghavi1/falcon/'+current_time+'/data.json5', 'w') as f:
    #     json5.dump(data, f, indent=4)

if __name__ == "__main__":
    main()
