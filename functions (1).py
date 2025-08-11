import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import tools21cm as t2c

def gaussimagegen(res=128, max_num_points=20, sigma=10):
    # Step 1: Create a 2D matrix of zeros
    matrix_size = (res, res)
    # matrix = np.zeros(matrix_size)

    # Step 2: Seed 10 points randomly with the value 1000
    # num_points = np.random.randint(1, max_num_points+1)
    # np.random.seed(42)  # For reproducibility
    # points = np.random.randint(0, matrix_size[0], size=(num_points, 2))


    matrix = torch.zeros(matrix_size, dtype=torch.float32)

    # Step 2: Seed up to `max_num_points` points randomly
    num_points = torch.randint(1, max_num_points + 1, (1,)).item()
    points = torch.randint(0, res, (num_points, 2))  # Random point coordinates

    # for point in points:
    #     matrix[point[0], point[1]] = 1
    # print(matrix)
    # Step 3: Apply different Gaussian kernels to each point
    # matrix = np.zeros(matrix_size)
    # print(matrix)
    # Step 3: Apply stronger Gaussian kernels to each point

    sigmas_x = torch.rand(num_points) * sigma
    sigmas_y = torch.rand(num_points) * sigma

    # Apply Gaussian kernel to each point
    for i, point in enumerate(points):
        sigma_x = sigmas_x[i].item()
        sigma_y = sigmas_y[i].item()
        matrix += apply_gaussian_kernel(matrix, point, sigma_x, sigma_y)

    return matrix

    # print(sigma_x)

    # sigmas_x = np.random.uniform(0, sigma, num_points)  # Increased range for more spread
    # sigmas_y = np.random.uniform(0, sigma, num_points)

    # # Apply Gaussian kernel to each point
    # for i, point in enumerate(points):
    #     sigma_x = sigmas_x[i]
    #     sigma_y = sigmas_y[i]
    #     matrix += apply_gaussian_kernel(matrix, point, sigma_x, sigma_y)
    
    # return matrix

def apply_gaussian_kernel(matrix, point, sigma_x, sigma_y):
        # Create a Gaussian kernel on an empty matrix
    gaussian_matrix = np.zeros_like(matrix.numpy())
    gaussian_matrix[point[0], point[1]] = 1
    return torch.tensor(gaussian_filter(gaussian_matrix, sigma=(sigma_x, sigma_y)), dtype=torch.float32)

# def sample_multivariate(n, mean, cov):
#     distribution = torch.distributions.MultivariateNormal(mean, cov)
#     samples = distribution.sample((n,))  # Generate n samples
#     return samples  # Returns an (n, 2) matrix

# Example usage:

# def simimage_from_uvmap(truimage, gain_uv_map):
#     truimagefour_coef = torch.fft.fft2(truimage)
#     gainsimimagefour_coef=truimagefour_coef * gain_uv_map
#     gainsimimagefour_coef = torch.nan_to_num(gainsimimagefour_coef, nan=0)
#     gainsimimagefour_coef = np.fft.fftshift(gainsimimagefour_coef)
#     gainsimimagefour_coef = torch.tensor(gainsimimagefour_coef)
#     # gainsimimage = torch.fft.ifft2(gainsimimagefour_coef)
#     # gainsimimage = np.real(gainsimimage)
#     # return gainsimimage, gainmatrix
#     return torch.real(gainsimimagefour_coef), torch.imag(gainsimimagefour_coef)


def simimage_from_uvmap(truimage, gain_uv_map_batch):
    """
    truimage: [H, W] complex tensor
    gain_uv_map_batch: [B, H, W] complex tensor
    Returns:
        real parts: [B, H, W]
        imag parts: [B, H, W]
    """

    # 1. FFT of truimage
    truimagefour_coef = torch.fft.fft2(truimage)  # [H, W]

    # 2. Expand to batch size
    batch_size = gain_uv_map_batch.shape[0]
    truimagefour_coef = truimagefour_coef.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H, W]

    # 3. Apply gain
    gainsimimagefour_coef = truimagefour_coef * gain_uv_map_batch  # [B, H, W]

    # 4. Handle NaNs
    gainsimimagefour_coef = torch.nan_to_num(gainsimimagefour_coef, nan=0.0)

    # 5. Shift FFT using np.fft.fftshift for each batch element
    # Convert to NumPy for fftshift
    # gainsimimagefour_coef_np = gainsimimagefour_coef.numpy()  # Convert to NumPy array

    # Apply np.fft.fftshift along the last two dimensions (height and width)
    # gainsimimagefour_coef_np = np.fft.fftshift(gainsimimagefour_coef_np, axes=(-2, -1))  # [B, H, W]

    # Convert back to PyTorch tensor
    # gainsimimagefour_coef = torch.tensor(gainsimimagefour_coef_np)

    # 6. Return real and imag parts
    return torch.real(gainsimimagefour_coef), torch.imag(gainsimimagefour_coef)
     
def getgainmap(gridsize, redshift, gain_model, gain_timescale=[10,60], subarray_type=None, total_int_time=6, int_time=10, boxsize=None, declination=-30., include_mirror_baselines=False, verbose=True):
    gain_uv_map, N_ant = t2c.get_uv_map_with_gains(
                        gridsize, redshift, 
                        gain_model,
                        gain_timescale, subarray_type,
                        total_int_time,
                        int_time,
                        boxsize, declination, include_mirror_baselines, verbose
                       )
    # print(gain_uv_map.shape)
    # print(gain_uv_map[:,:,1][gain_uv_map[:, :, 1] != 0])
    finalgainuvmap = (gain_uv_map[:,:,1] + gain_uv_map[:,:,2]+(-gain_uv_map[:,:,3]+gain_uv_map[:,:,4])*1j)/gain_uv_map[:,:,0]
    finalgainuvmap = torch.tensor(np.nan_to_num(finalgainuvmap))
    return finalgainuvmap

def build_features_from_gain_model(gain_model_batch,truimage,gridsize,redshift,gain_timescale,subarray_type,total_int_time,int_time,box_len, noise = 0):
    """
    Constructs input features from a batch of gain models.
    
    Parameters:
    - gain_model_batch: Tensor of shape (batch_size, 8)
    - truimage: Reference image
    - getgainmap_fn: function to generate gain uv maps
    - simimage_from_uvmap_fn: function to simulate image from uv map
    - All other parameters: used to call getgainmap
    
    Returns:
    - x: Tensor of shape (batch_size, 2 * num_nonzero_pixels + 3)
    """
    batch_size = gain_model_batch.shape[0]

    # Reshape: (batch_size, 8) → (batch_size, 4, 2)
    tensor1 = gain_model_batch.view(batch_size, -1, 2)

    finalgainuvmap_all = []
    for i in range(batch_size):
        gain_uv_map = getgainmap(
            gridsize=gridsize,
            redshift=redshift,
            gain_model=tensor1[i],
            gain_timescale=gain_timescale,
            subarray_type=subarray_type,
            total_int_time=total_int_time,
            int_time=int_time,
            boxsize=box_len
        )
        # print(gain_uv_map[:,:,1][gain_uv_map[:, :, 1] != 0].shape)
        finalgainuvmap_all.append(gain_uv_map)

    finalgainuvmap_all = torch.stack(finalgainuvmap_all, dim=0)

    # Simulate image
    simresr, simresi = simimage_from_uvmap(truimage, finalgainuvmap_all)

    # Flatten and apply non-zero mask
    simresr_flat = simresr.view(batch_size, -1)
    simresi_flat = simresi.view(batch_size, -1)
    mask = simresr_flat[0] != 0  # shared mask
    simr = simresr_flat[:, mask]
    simi = simresi_flat[:, mask]
    simr = simr + noise * torch.randn_like(simr)
    simi = simi + noise * torch.randn_like(simi)
    # Get gain info from g1
    complex_tensor = torch.view_as_complex(tensor1)
    phase1 = torch.angle(complex_tensor[:, 0]).unsqueeze(1)
    amp1 = torch.abs(complex_tensor[:, 0]).unsqueeze(1)

    # Final concatenation
    x = torch.cat((simr, simi, torch.sin(phase1), torch.cos(phase1), amp1), dim=1)
    return x

# @torch.jit.script
def makeuvmapnew(paddedtensor, subsetindices, baselinegain, H, W):

    """
    Generate a time-weighted UV map tensor from baseline positions and gains.

    This function constructs a UV map for a given subset of baselines by 
    scattering time values weighted by baseline gains onto a 2D spatial grid.

    Parameters
    ----------
    paddedtensor : torch.Tensor
        Tensor of shape (N, 3, L) representing baseline positions and times.
        For each baseline:
            - Index 0 corresponds to x-coordinates (pixels).
            - Index 1 corresponds to y-coordinates (pixels).
            - Index 2 corresponds to times (or time-related values).
    subsetindices : torch.Tensor or list
        1D tensor or list of indices selecting a subset of baselines from `paddedtensor`.
    baselinegain : torch.Tensor
        Tensor of shape (Batchsize, N) representing gain weights for each baseline
        across a batch of samples.
    H : int
        Height of the output UV map (number of pixels vertically).
    W : int
        Width of the output UV map (number of pixels horizontally).

    Returns
    -------
    output : torch.Tensor
        Tensor of shape (Batchsize, H, W), where each spatial position (x, y) 
        contains the sum of time values weighted by baseline gains, scattered 
        across the UV map for each batch element. Note that if a baseline was observed at 1,127, the image will be correct because matplotlib
        has a flipped y system, but the print(matrix) will show value at 127,1 and the values are all stored in 127,1. This is because the
        baseline is stored as (x,y) and the image is stored as (y,x).
    """
    device = baselinegain.device
    dtype = baselinegain.dtype

    # Shapes
    # baseposnlist: (N, 3, L)
    # indices: (B,)
    # output: (B, H, W)
    # B = subsetindices.shape[0]
    Batchsize=baselinegain.shape[0]
    # B = subsetindices.shape[0]  # Number of baselines in the subset
    # L = paddedtensor.shape[2]  # Length of time dimension
    selected = paddedtensor[subsetindices]     # shape (B, 3, L)
    # x =                 # shape (B, L)
    # y =                 # shape (B, L)
    t = selected[:, 2, :]                # shape (B, L)
    bg =  baselinegain[:, subsetindices]  # shape (Batch, B)
    
    # bg =                # (Batchsize, B, 1)
    # t =                # (1, B, L)
    # weighted_t = bg.unsqueeze(-1) * t.unsqueeze(0)              # (Batchsize, B, L)
    # print(weighted_t)  # Should be (Batchsize, B, L)
    # weighted_t = weighted_t.sum(dim=1)  # Sum over B, shape (Batchsize, L)
    weighted_t = torch.einsum('bi,il->bil', bg, t)  # → shape: (Batchsize, B, L)
    # print(weighted_t)  # Should be (Batchsize, L)
    # Compute flattened spatial indices
    flat_idx = (selected[:, 1, :] * W + selected[:, 0, :]).long()           # (B, L)
    # Use flat_idx from one baseline (all B have same spatial shape)
    # Flatten for vectorized scatter_add_
    # flat_idx = flat_idx  # (Batchsize, B, L)
    # weighted_t = weighted_t  # already (Batchsize, B, L)
    # Reshape to (Batchsize * B, L) for scatter_add_

    # Expand for batch: (Batchsize, B, L)
    flat_idx = flat_idx.unsqueeze(0).expand(Batchsize, -1, -1)
    flat_idx = flat_idx.view(Batchsize, -1)        # (Batchsize, B * L)
    weighted_t = weighted_t.view(Batchsize, -1)    # (Batchsize, B * L)

    # Prepare output and scatter
    output = torch.zeros(Batchsize, H * W, device=device, dtype=dtype)
    output.scatter_add_(1, flat_idx, weighted_t)

    return output.view(Batchsize, H, W).transpose(1, 2)
    
def baselinefromgain(gainlists):
    batch_size = gainlists.shape[0]
    nant = gainlists.shape[1]//2
    # Reshape real tensor from (batch_size, 8) → (batch_size, 4, 2) for complex view
    gain_real_imag = gainlists.view(batch_size, -1, 2)

    # Convert pairs of real/imag parts to complex numbers (batch_size, 4)
    gain_complex = torch.view_as_complex(gain_real_imag)
    first_complex = gain_complex[:, 0]

    # Compute magnitude and phase
    real = torch.real(first_complex)  # |z|
    imag = torch.imag(first_complex)    # angle(z)
    idx_i, idx_j = torch.triu_indices(nant, nant, offset=1)  # each shape: (num_pairs,)

    # 2. Select pairs from gainlist
    # gainlist[:, idx_i] shape: (B, num_pairs)
    # gainlist[:, idx_j] shape: (B, num_pairs)
    baselinegains = gain_complex[:, idx_i] * torch.conj(gain_complex[:, idx_j])  # shape: (B, num_pairs)
    return baselinegains, real, imag

def sampler12a(fourimag, paddedtensor, subsetindiceslist, gainlist, H, W, noiselevel):
    nant = gainlist.shape[1]//2
    baselinegain,real1st,imag1st = baselinefromgain(gainlist)
    # sin_phase = torch.sin(angle)  # shape: (batch_size,)
    # cos_phase = torch.cos(angle)  # shape: (batch_size,)
    subuvmaps = []

    for subsetindex in subsetindiceslist:
        subuvmap = makeuvmapnew(paddedtensor, subsetindex, baselinegain, H, W)  # (B, H, W)
        subuvmaps.append(subuvmap.unsqueeze(1))  # (B, 1, H, W)
    # Stack into (B, 2, H, W)
    stacked = torch.cat(subuvmaps, dim=1)
    # print(stacked.shape)
    # print(fourimag.shape)
    output = stacked*fourimag.unsqueeze(1)
    output = output + torch.randn_like(output) * noiselevel
    outputreal = torch.real(output)
    outputimag = torch.imag(output)
    return outputreal, outputimag, real1st, imag1st

def pointimagecreator(x,y,xsize,ysize):
    image = torch.zeros(xsize,ysize)
    image[x,y] = 1
    return image

def interleave_ri(n):
    r = [f"r{i+1}" for i in range(n)]  # Replace with actual values if needed
    i = [f"i{i+1}" for i in range(n)]  # Replace with actual values if needed
    return [val for pair in zip(r, i) for val in pair]

def split_indices(n, k):
    indices = torch.arange(n)
    base_size = n // k
    remainder = n % k

    sizes = torch.full((k,), base_size)
    sizes[:remainder] += 1

    starts = torch.cat((torch.tensor([0]), torch.cumsum(sizes, dim=0)[:-1]))

    splits = [indices[start:start + size] for start, size in zip(starts, sizes)]
    return splits







#PS+source
import torch.fft

### --- Sampling code from earlier ---
def make_kgrid(shape, BoxSize, device):
    ny, nx = shape
    dkx = 2 * np.pi / BoxSize
    dky = 2 * np.pi / BoxSize
    ky = torch.fft.fftfreq(ny, device=device) * ny * dky
    kx = torch.fft.fftfreq(nx, device=device) * nx * dkx
    kx, ky = torch.meshgrid(kx, ky, indexing='xy')
    k = torch.sqrt(kx**2 + ky**2)
    return k


def power_spectrum_2d(k, A, n):
    k = torch.where(k == 0, torch.tensor(1.0, device=k.device), k)
    A = A[:, None, None]
    n = n[:, None, None]
    return A * k**n

# Generate complex noise with Hermitian symmetry
def generate_hermitian_noise(B, H, W, device):
    real_part = torch.randn((B, H, W//2 + 1), device=device)
    imag_part = torch.randn((B, H, W//2 + 1), device=device)
    imag_part[..., 0] = 0  # Nyquist mode is real
    if W % 2 == 0:
        imag_part[..., -1] = 0

    complex_noise = real_part + 1j * imag_part
    return complex_noise

# Use rfft2 / irfft2 to preserve Hermitian structure
def sample_density_field(A, n, batchsize=None, shape=128, BoxSize=1.0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    if batchsize is None:
        B = A.shape[0]
    else:
        B = batchsize
    device = A.device
    H = shape
    W = shape
    N = H
    dx = BoxSize / N

    kx = torch.fft.rfftfreq(W, d=1./W).to(device) * 2 * np.pi / BoxSize
    ky = torch.fft.fftfreq(H, d=1./H).to(device) * 2 * np.pi / BoxSize
    kx, ky = torch.meshgrid(kx, ky, indexing='xy')
    k = torch.sqrt(kx**2 + ky**2)  # [H, W//2+1]

    pk = power_spectrum_2d(k, A, n) * dx**4  # [B, H, W//2+1]
    noise = generate_hermitian_noise(B, H, W, device) * torch.sqrt(pk / 2)

    # inverse rFFT to real field
    field = torch.fft.irfft2(noise, s=(H, W), norm='ortho')
    field = field/field.mean(dim=(1, 2), keepdim=True) -1
    return field

def brightnesstempfield(densityfield, ionizationfraction=1, z=9, TCMB=0, TS=1, omegab=0.044, h=0.7, omegam=0.27, yp=0.248):
    brightnesstempfield = 27*ionizationfraction*(1+densityfield)*(((1+z)/10)**0.5)*(1-TCMB/TS)*(omegab/0.044)*(h/0.7)*((omegam/0.27)**-0.5)*((1-yp)/(1-0.248))
    return brightnesstempfield

def noisefieldfunc(nu=1420.4, nuc=110, z=9, area=962, deltanu=0.085, deltat=10, H=128, W=128):
    nu=nu/(1+z)
    Tsys = 60*((300/nu)**2.55)
    e = (nuc/nu)**2
    sigma = np.sqrt(2)*1.380649e-23*Tsys/(e*area*np.sqrt(deltanu*deltat*1e6))
    return sigma

def compute_2d_power_spectrum(field, BoxSize, nbins=50):
    B, H, W = field.shape
    dx = BoxSize / H

    # FFT and power spectrum
    fft = torch.fft.fft2(field, norm='ortho')
    power = (fft.real ** 2 + fft.imag ** 2) * dx**4  # [B, H, W]

    # k-grid
    kgrid = make_kgrid((H, W), BoxSize, field.device).flatten()  # [H*W]
    power_flat = power.view(B, -1)  # [B, H*W]

    # Bin edges and centers
    kmin = kgrid[kgrid > 0].min()
    kmax = kgrid.max()
    k_edges = torch.linspace(kmin, kmax, nbins + 1, device=field.device)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])  # [nbins]

    # Bin indices
    bin_indices = torch.bucketize(kgrid, k_edges) - 1  # [H*W], 0-based

    # Valid bins only
    valid = (bin_indices >= 0) & (bin_indices < nbins)
    bin_indices = bin_indices[valid]  # [Nvalid]
    power_flat = power_flat[:, valid]  # [B, Nvalid]

    # One-hot encode bin indices: [Nvalid, nbins]
    one_hot = torch.nn.functional.one_hot(bin_indices, nbins).float()

    # Count modes per bin: [nbins]
    Nmodes = one_hot.sum(dim=0)

    # Weighted sum: [B, nbins]
    Pk_sum = power_flat @ one_hot  # matmul [B, Nvalid] x [Nvalid, nbins]

    # Avoid divide-by-zero
    Pk = torch.where(Nmodes > 0, Pk_sum / Nmodes, torch.zeros_like(Pk_sum))

    return k_centers, Pk, Nmodes

def gaussian_2d(size=128, sigma=20):
    # Create coordinate vectors centered at zero
    coords = torch.linspace(-(size - 1) / 2, (size - 1) / 2, steps=size)
    x, y = torch.meshgrid(coords, coords, indexing='xy')

    # Compute 2D Gaussian
    gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    return gaussian

def sampler13(theta, an, shape, image, paddedtensor, subsetindiceslist, noise, nu, z, res=7):
    theta[:, 0::2] += 1  # Add 1 to all real parts
    # Create the pair (1, 0) and repeat it for each batch element
    prepend_pair = torch.tensor([1.0, 0.0]).unsqueeze(0).repeat(theta.size(0), 1)
    # Concatenate along the last dimension
    theta = torch.cat([prepend_pair, theta], dim=1)
    a = an[:,0]
    n = an[:,1]
    field = sample_density_field(a, n, batchsize=None, shape = shape[0], BoxSize=shape[0], seed=42)
    field = brightnesstempfield(field)
    image_obs = field+image*10000
    uv_obs = torch.fft.fft2(image_obs)
    unitimage=torch.ones((1, shape[0], shape[1]))
    unitgain = torch.arange(theta.shape[1]) % 2   # 0,1,0,1,...
    unitgain = 1 - unitgain             # 1,0,1,0,...
    unitgain = unitgain.unsqueeze(0).float()   # shape: (1, 128)
    unitimagecountreal,  unitimagecountimag, real1st, imag1st = sampler12a(fourimag=unitimage, paddedtensor = paddedtensor, subsetindiceslist=subsetindiceslist, gainlist=unitgain, H=shape[0], W=shape[1], noiselevel=0)
    outputsummedreal, outputsummedimag, real1st, imag1st = sampler12a(fourimag=uv_obs, paddedtensor = paddedtensor, subsetindiceslist=subsetindiceslist, gainlist=theta, H=shape[0], W=shape[1], noiselevel=0)
    noisesigma = noisefieldfunc(nu=nu, nuc=110, z=z, area=962, deltanu=0.085, deltat=10, H=shape[0], W=shape[1])
    resrad=res*np.pi/180/3600
    noisesigma = noisesigma*(299792458**2)*((1+z)**2)/(2*nu*nu*1.380649e-23*1.1331*resrad*resrad*1e12)*1000*noise
    outputsummedreal = outputsummedreal/unitimagecountreal
    outputsummedimag = outputsummedimag/unitimagecountreal
    outputsummedreal = torch.nan_to_num(outputsummedreal, nan=0.0)
    outputsummedimag = torch.nan_to_num(outputsummedimag, nan=0.0)
    outputnorm = torch.complex(outputsummedreal, outputsummedimag)#+noisefield
    noisefield = torch.randn_like(outputnorm)*noisesigma*noise
    noisefield = noisefield/torch.sqrt(unitimagecountreal)
    noisefield = torch.nan_to_num(noisefield, nan=0.0)
    return outputnorm+noisefield