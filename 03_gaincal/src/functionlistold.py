import torch
import tools21cm as t2c
import numpy as np

def thetatogains(theta: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor of theta values into a gains tensor by prepending (1, 0) 
    and incrementing all real parts by 1.
    Args:
        theta (torch.Tensor): A 2D tensor of shape (batch_size, 2 * (n-1)), where the 
                              real and imaginary parts alternate along the last dimension.
                              Example shape: [batch_size, 2 * (n-1))]. n generally is the
                              number of antennae in this case

    Returns:
        torch.Tensor: A 2D tensor of shape (batch_size, 2 * (n)), where the first
                      pair is always (1, 0), followed by the modified theta values.

    Example:
        >>> theta = torch.tensor([[0.5, 0.2, 0.7, 0.3]])
        >>> thetatogains(theta)
        tensor([[1.0000, 0.0000, 1.5000, 0.2000, 1.7000, 0.3000]])
    """
    batch_size, features = theta.shape
    # Pre-allocate gains with an extra complex pair
    gains = torch.empty((batch_size, features + 2),
                        dtype=theta.dtype,
                        device=theta.device)
    # First, set the prepend pair to (0, 0)
    gains[:, 0:2] = 0.0
    # Copy theta into gains[:, 2:]
    gains[:, 2:] = theta
    # Add +1 to **all** real parts, including the prepended one
    gains[:, 0::2] += 1.0
    return gains

def baselinefromgain(gainlists: torch.Tensor):
    """
    Compute baseline gains from per-antenna complex gains.

    Given a batch of per-antenna gains (stored as interleaved real/imag values), 
    this function:
    1. Converts them into complex numbers.
    2. Computes the pairwise baseline gains for all antenna pairs (i, j).
    3. Returns baseline gains

    Args:
        gainlists (torch.Tensor): A 2D tensor of shape (B, 2 * N), where:
            - B = batch size.
            - N = number of antennas.
            - Gains are stored as [Re(g0), Im(g0), Re(g1), Im(g1), ..., Re(gN-1), Im(gN-1)].

    Returns:
            - baselinegains (torch.Tensor): Complex baseline gains, shape (B, P),
              where P = N*(N-1)/2 (number of unique antenna pairs).

    Example:
        >>> gains = torch.tensor([[1.0, 0.0, 0.5, 0.5, -0.5, 0.5, 1.0, -1.0]])  # B=1, N=4
        >>> baselinegains = baselinefromgain(gains)
        >>> baselinegains.shape
        torch.Size([1, 6])  # 4 antennas → 6 baselines
    """
    batch_size = gainlists.shape[0]
    nant = gainlists.shape[1] // 2  # Number of antennas
    # Reshape real/imag pairs from (B, 2*N) → (B, N, 2)
    gain_real_imag = gainlists.view(batch_size, nant, 2)
    # Convert pairs of real/imag parts to complex numbers → shape: (B, N)
    gain_complex = torch.view_as_complex(gain_real_imag)
    # Compute all unique antenna index pairs (upper triangular, excluding diagonal)
    idx_i, idx_j = torch.triu_indices(nant, nant, offset=1)
    # Compute baseline gains: G_i * conj(G_j), shape → (B, P)
    baselinegains = gain_complex[:, idx_i] * torch.conj(gain_complex[:, idx_j])
    return baselinegains

class subsetUVMapGenerator:
    """
    subset UV Map Generator for baseline-weighted time accumulation.
    Precomputes fixed geometry.
    """
    def __init__(self, paddedtensor: torch.Tensor, subsetindices: torch.Tensor, H: int, W: int):
        """
        Initialize the UV Map Generator.

        Parameters
        ----------
        paddedtensor : torch.Tensor
            Tensor of shape (N, 3, L) representing baseline positions and times.
        subsetindices : torch.Tensor or list
            Indices selecting a subset of baselines.
        H : int
            Height of the UV map.
        W : int
            Width of the UV map.
        """
        # Move fixed data to GPU if possible
        device = paddedtensor.device
        self.device = device
        self.dtype = paddedtensor.dtype
        self.paddedtensor = paddedtensor
        self.subsetindices = torch.as_tensor(subsetindices, device=device, dtype=torch.long)
        self.H = H
        self.W = W
        # Precompute selected subset → shape: (B, 3, L)
        self.selected = self.paddedtensor[self.subsetindices]
        # Extract x, y, t → shapes: (B, L)
        self.x = self.selected[:, 0, :].to(device)
        self.y = self.selected[:, 1, :].to(device)
        self.t = self.selected[:, 2, :].to(device)
        # Precompute flattened spatial indices → shape: (B, L)
        self.flat_idx_template = (self.y * W + self.x).long()

    def __call__(self, baselinegain: torch.Tensor) -> torch.Tensor:
        """
        Generate a subset time-weighted UV map for a given batch of baseline gains.

        Parameters
        ----------
        baselinegain : torch.Tensor
            Tensor of shape (Batchsize, N) representing gain weights
            for each baseline across the batch.

        Returns
        -------
        torch.Tensor
            UV map of shape (Batchsize, H, W), where each pixel contains
            the sum of time values weighted by baseline gains.
        """
        # Ensure baselinegain is on the same device and dtype
        baselinegain = baselinegain.to(self.device)
        Batchsize = baselinegain.shape[0]
        # Select relevant baseline gains → shape: (Batchsize, B)
        bg = baselinegain[:, self.subsetindices]  # (Batchsize, B)
        # Weighted time values → (Batchsize, B, L)
        weighted_t = torch.einsum('bi,il->bil', bg, self.t.to(baselinegain.dtype))
        # Expand flat indices for batch → (Batchsize, B, L)
        flat_idx = self.flat_idx_template.unsqueeze(0).expand(Batchsize, -1, -1)
        # Reshape for scatter → (Batchsize, B*L)
        flat_idx = flat_idx.reshape(Batchsize, -1)
        weighted_t = weighted_t.reshape(Batchsize, -1)
        # Preallocate output on GPU
        output = torch.zeros(Batchsize, self.H * self.W, device=self.device, dtype=baselinegain.dtype)
        # Use scatter_add_, which is CUDA-optimized
        output.scatter_add_(1, flat_idx, weighted_t)
        # Reshape and fix orientation for visualization
        return output.view(Batchsize, self.H, self.W).transpose(1, 2)

class UVMapGenerator:
    """
    UVMapGenerator: Precomputes and stacks UV maps using subsetUVMapGenerator.
    Designed to return UV maps only, so that fourimag multiplication
    can be done outside.
    """
    def __init__(self, paddedtensor, subsetindiceslist, H, W):
        """
        Initialize the sampler.

        Parameters
        ----------
        paddedtensor : torch.Tensor
            Baseline position/time tensor, shape (N, 3, L).
        subsetindiceslist : list[torch.Tensor]
            List of index tensors selecting subsets of baselines.
        H : int
            Height of UV map.
        W : int
            Width of UV map.
        """
        self.paddedtensor = paddedtensor
        self.subsetindiceslist = subsetindiceslist
        self.H = H
        self.W = W
        self.device = paddedtensor.device
        # Pre-initialize UVMapGenerators for each subset to avoid recomputation
        self.uvgens = [
            subsetUVMapGenerator(paddedtensor, subset, H, W)
            for subset in subsetindiceslist
        ]

    def __call__(self, gainlist):
        """
        Generate stacked UV maps for a given gainlist.

        Parameters
        ----------
        gainlist : torch.Tensor
            Antenna gains of shape (B, 2 * nant), possibly complex.

        Returns
        -------
        stacked : torch.Tensor
            Stacked UV maps of shape (B, num_subsets, H, W).
        """
        # Convert antenna gains to baseline gains
        baselinegain = baselinefromgain(gainlist)  # baselinegain is complex
        # Generate UV maps for each subset using cached UVMapGenerators
        subuvmaps = []
        for uvgen in self.uvgens:
            subuvmap = uvgen(baselinegain)  # shape: (B, H, W)
            subuvmaps.append(subuvmap.unsqueeze(1))  # shape: (B, 1, H, W)
        # Stack UV maps → shape: (B, num_subsets, H, W)
        stacked = torch.cat(subuvmaps, dim=1)
        return stacked

def multiply_uvmaps_with_fourier(stacked, fourimag):
    """
    Multiply stacked UV maps with Fourier image.

    Parameters
    ----------
    stacked : torch.Tensor
        Stacked UV maps, shape (B, num_subsets, H, W).
    fourimag : torch.Tensor
        Fourier image, shape (B, H, W).

    Returns
    -------
    output : torch.Tensor
        Resulting tensor, shape (B, num_subsets, H, W).
    """
    if fourimag.dim() == 3:
        fourimag = fourimag.unsqueeze(1)  # (B, 1, H, W)
    return stacked * fourimag

class UnitUVMapGenerator:
    """
    Precompute and store stacked UV maps for default antenna gains (all ones: 1+0j).
    """
    def __init__(self, paddedtensor, subsetindiceslist, H, W, nant, device=None, dtype=torch.float32):
        """
        Initialize and precompute stacked UV maps.

        Parameters
        ----------
        paddedtensor : torch.Tensor
            Baseline position/time tensor of shape (N, 3, L).
        subsetindiceslist : list[torch.Tensor]
            List of index tensors selecting subsets of baselines.
        H : int
            Height of UV map.
        W : int
            Width of UV map.
        nant : int
            Number of antennas.
        device : torch.device, optional
            Device to store the result (default: same as paddedtensor).
        dtype : torch.dtype, optional
            Data type for the precomputed maps (default: torch.float32).
        """
        if device is None:
            device = paddedtensor.device

        self.device = device
        self.H = H
        self.W = W
        self.subsetindiceslist = subsetindiceslist
        self.nant = nant
        # Create default gainlist: interleaved real/imag = [1,0,1,0,...]
        unittheta = torch.zeros((1, 2 * (nant-1)), device=device, dtype=dtype)
        unitgains = thetatogains(unittheta)
        # Initialize UVMapGenerator for all subsets
        uvgen = UVMapGenerator(paddedtensor, subsetindiceslist, H, W)
        # Compute and store stacked UV maps
        self.stacked = uvgen(unitgains)  # shape: (1, num_subsets, H, W)

    def __call__(self):
        """
        Return the precomputed stacked UV maps.

        Returns
        -------
        torch.Tensor
            Stacked UV maps of shape (1, num_subsets, H, W)
        """
        return self.stacked

# ===============================
# Class to hold simulation parameters
# ===============================
class SimulationConfig:
    """
    Stores all configuration parameters for the 2D/3D 21-cm brightness temperature simulation.
    This acts as a single source of truth for grid, cosmological, and Gaussian parameters.
    """

    def __init__(
        self,
        # Grid properties
        shape=128,
        BoxSize=128.0,

        # Gaussian noise properties
        sigma=10.0,
        prefactor = 27.0,
        # Cosmological & physical parameters
        ionizationfraction=1.0,  # x_HI
        z=9.0,                  # Redshift
        TCMB=0.0,               # CMB temperature [K]
        TS=1.0,                 # Spin temperature [K]
        omegab=0.044,           # Ω_b: baryon density
        h=0.7,                  # Hubble parameter (H0 = 100 h km/s/Mpc)
        omegam=0.27,            # Ω_m: total matter density
        yp=0.248,               # Helium mass fraction Y_p

        # Device configuration
        device=None
    ):
        # Grid & box properties
        self.shape = shape
        self.BoxSize = BoxSize
        self.H = self.W = shape
        self.dx = BoxSize / shape

        # Gaussian settings
        self.sigma = sigma

        # Cosmological parameters
        self.ionizationfraction = ionizationfraction
        self.z = z
        self.TCMB = TCMB
        self.TS = TS
        self.omegab = omegab
        self.h = h
        self.omegam = omegam
        self.yp = yp
        self.prefactor = prefactor
        # Device auto-selection (CUDA if available)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def summary(self):
        """
        Print a summary of all simulation parameters for debugging.
        """
        print("\n===== Simulation Configuration =====")
        print(f"Grid shape          : {self.shape} x {self.shape}")
        print(f"Box size            : {self.BoxSize} Mpc")
        print(f"dx                  : {self.dx:.3f} Mpc")
        print(f"Device              : {self.device}")
        print("\n--- Cosmological Parameters ---")
        print(f"Redshift z          : {self.z}")
        print(f"Spin temperature TS : {self.TS} K")
        print(f"CMB temperature     : {self.TCMB} K")
        print(f"Ω_b (baryons)       : {self.omegab}")
        print(f"Ω_m (matter)        : {self.omegam}")
        print(f"h (Hubble param)    : {self.h}")
        print(f"Y_p (helium frac)   : {self.yp}")
        print(f"Ionization fraction : {self.ionizationfraction}")
        print("\n--- Gaussian Settings ---")
        print(f"Sigma               : {self.sigma}")
        print(f"Prefactor           : {self.prefactor}")
        print("====================================\n")

# ===============================
# Utility functions
# ===============================
# ===============================
# Utility functions
# ===============================
def make_kgrid(shape, BoxSize, device, rfft=False):
    """
    Generate a 2D wavenumber grid (kx, ky) magnitude.

    Args:
        shape (tuple): (H, W) grid dimensions.
        BoxSize (float): physical size of the box.
        device: torch device.
        rfft (bool): if True, kx is for rFFT (size W//2+1).

    Returns:
        torch.Tensor: k = sqrt(kx^2 + ky^2), shape [H, W] or [H, W//2+1]
    """
    H, W = shape
    dx, dy = BoxSize / H, BoxSize / W

    kx = torch.fft.rfftfreq(W, d=dy).to(device) * 2*np.pi if rfft else torch.fft.fftfreq(W, d=dy).to(device) * 2*np.pi
    ky = torch.fft.fftfreq(H, d=dx).to(device) * 2*np.pi
    kx, ky = torch.meshgrid(kx, ky, indexing='xy')
    return torch.sqrt(kx**2 + ky**2)

def power_spectrum_2d(k, A, n):
    """
    Compute a 2D power spectrum of the form P(k) = A * k^n.

    Args:
        k (torch.Tensor): 2D wavenumber grid.
        A (torch.Tensor): Amplitude(s), shape [B]
        n (torch.Tensor): Spectral index(s), shape [B]

    Returns:
        torch.Tensor: P(k), shape [B, H, W]
    """
    k_safe = torch.where(k == 0, torch.tensor(1.0, device=k.device), k)
    return A[:, None, None] * k_safe**n[:, None, None]

def generate_hermitian_noise(B, H, W, device):
    """
    Generate Hermitian Gaussian noise for Fourier-space field.

    Args:
        B (int): batch size.
        H, W (int): grid dimensions.
        device: torch device.

    Returns:
        torch.Tensor: complex Hermitian noise [B, H, W//2+1]
    """
    real = torch.randn((B, H, W//2+1), device=device)
    imag = torch.randn((B, H, W//2+1), device=device)
    imag[..., 0] = 0.0  # Nyquist real
    if W % 2 == 0:
        imag[..., -1] = 0.0
    return real + 1j * imag

# ===============================
# Main simulator class
# ===============================
class DensityField2D:
    """
    Generates 2D Gaussian random density fields and computes power spectra.
    """
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.H = config.H
        self.W = config.W
        self.BoxSize = config.BoxSize
        self.dx = config.dx
        self.device = config.device

    def sample_field(self, A, n):
        """
        Sample 2D density field δ = ρ/ρ̄ - 1 using a given P(k).

        Args:
            A (torch.Tensor): amplitudes [B]
            n (torch.Tensor): spectral indices [B]

        Returns:
            delta (torch.Tensor): 2D density field [B, H, W]
        """
        B = A.shape[0]
        # Fourier-space grid
        k = make_kgrid((self.H, self.W), self.BoxSize, self.device, rfft=True)

        # Power spectrum scaled by dx^2
        pk = power_spectrum_2d(k, A, n) * self.dx**2

        # Generate Hermitian Gaussian noise
        noise = generate_hermitian_noise(B, self.H, self.W, self.device) * torch.sqrt(pk / 2)
        noise[..., 0, 0] = 0.0  # remove DC mode

        # Inverse rFFT
        delta = torch.fft.irfft2(noise, s=(self.H, self.W), norm='ortho')
        return delta

    def compute_power_spectrum(self, field, nbins=50):
        """
        Compute the isotropic 2D power spectrum from a field.

        Args:
            field (torch.Tensor): [B, H, W] real-space density field.
            nbins (int): number of radial k bins.

        Returns:
            k_centers (torch.Tensor): bin centers
            Pk (torch.Tensor): estimated P(k) [B, nbins]
            Nmodes (torch.Tensor): number of modes per bin
        """
        B, Hf, Wf = field.shape
        assert Hf == self.H and Wf == self.W, "Field shape mismatch"

        # FFT and power
        fft = torch.fft.fft2(field, norm='ortho')
        power = (fft.real**2 + fft.imag**2) * self.dx**-2

        # Flatten k-grid and power
        kgrid = make_kgrid((self.H, self.W), self.BoxSize, self.device, rfft=False).flatten()
        power_flat = power.view(B, -1)

        # Bin k values
        kpos = kgrid[kgrid > 0]
        kmin, kmax = kpos.min(), kgrid.max()
        k_edges = torch.linspace(kmin, kmax, nbins+1, device=self.device)
        k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

        bin_indices = torch.bucketize(kgrid, k_edges) - 1
        valid = (bin_indices >= 0) & (bin_indices < nbins)
        bin_indices = bin_indices[valid]
        power_flat = power_flat[:, valid]

        # One-hot binning
        one_hot = torch.nn.functional.one_hot(bin_indices, nbins).float()
        Nmodes = one_hot.sum(dim=0)
        Pk_sum = power_flat @ one_hot
        Pk = torch.where(Nmodes > 0, Pk_sum / Nmodes, torch.zeros_like(Pk_sum))

        return k_centers, Pk, Nmodes

class BrightnessTemperature:
    """
    Computes the 21-cm brightness temperature field (δT_b) from a given density field.

    Formula:
        δT_b = 27 * x_HI * (1 + δ) * sqrt((1 + z)/10) * (1 - T_CMB/T_S)
               * (Ω_b/0.044) * (h/0.7) * (Ω_m/0.27)^(-0.5) * ((1 - Y_p)/(1 - 0.248))

    Parameters
    ----------
    z : float, optional
        Redshift of observation (default: 9).
    TCMB : float, optional
        Cosmic Microwave Background (CMB) temperature [K] (default: 0).
    TS : float, optional
        Spin temperature [K] (default: 1).
    omegab : float, optional
        Baryon density parameter Ω_b (default: 0.044).
    h : float, optional
        Dimensionless Hubble parameter (default: 0.7).
    omegam : float, optional
        Total matter density parameter Ω_m (default: 0.27).
    yp : float, optional
        Helium mass fraction Y_p (default: 0.248).

    Usage Example
    -------------
    >>> bt = BrightnessTemperature(z=8, TS=100)
    >>> delta = torch.randn((128, 128))  # density contrast δ = ρ/ρ̄ - 1
    >>> Tb = bt(delta, ionizationfraction=1.0)
    """

    def __init__(self, ionizationfraction=1.0, z=9.0, TCMB=0, TS=1, omegab=0.044, h=0.7, omegam=0.27, yp=0.248):
        # Cosmological & physical parameters
        self.z = z
        self.TCMB = TCMB
        self.TS = TS
        self.omegab = omegab
        self.h = h
        self.omegam = omegam
        self.yp = yp
        self.ionizationfraction=ionizationfraction

    def __call__(self, density_field, ):
        """
        Compute the 21-cm brightness temperature field δT_b.

        Parameters
        ----------
        density_field : torch.Tensor
            The 2D or 3D overdensity field δ = ρ/ρ̄ - 1.
        ionizationfraction : float or torch.Tensor, optional
            Neutral hydrogen fraction x_HI (default: 1.0).

        Returns
        -------
        torch.Tensor
            Brightness temperature field δT_b [mK].
        """
        Tb = (
            27.0
            * self.ionizationfraction
            * (1 + density_field)
            * (((1 + self.z) / 10) ** 0.5)
            * (1 - self.TCMB / self.TS)
            * (self.omegab / 0.044)
            * (self.h / 0.7)
            * ((self.omegam / 0.27) ** -0.5)
            * ((1 - self.yp) / (1 - 0.248))
        )
        return Tb

def gaussian_2d(size=128, sigma=10.0):
    """
    Generate a 2D Gaussian kernel using PyTorch.

    Parameters
    ----------
    size : int, optional
        Size of the Gaussian kernel (size x size). Default: 128.
    sigma : float, optional
        Standard deviation of the Gaussian. Controls the spread. Default: 20.0.

    Returns
    -------
    torch.Tensor
        2D Gaussian tensor of shape `(size, size)`.

    Example
    -------
    >>> g = gaussian_2d(size=128, sigma=15)
    >>> g.shape
    torch.Size([128, 128])
    """
    # Auto-select device
    # device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Create coordinate vectors centered at zero
    coords = torch.linspace(-(size - 1) / 2, (size - 1) / 2, steps=size)
    x, y = torch.meshgrid(coords, coords, indexing='xy')
    # Compute 2D Gaussian
    gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    # # Normalize if requested
    # gaussian = gaussian / gaussian.sum()
    return gaussian

import torch

def combine_brightness_with_gaussian(delta, config, gaussian_image=None):
    """
    Combine 21-cm brightness temperature with an optional Gaussian/noise image.

    Parameters
    ----------
    delta : torch.Tensor
        Overdensity field δ = ρ/ρ̄ - 1, shape [B, H, W].
    config : SimulationConfig
        Simulation configuration containing cosmology, shape, device, and prefactor.
    gaussian_image : torch.Tensor, optional
        Gaussian image to add, shape [H, W] or [B, H, W]. If None, a zero tensor is used.

    Returns
    -------
    torch.Tensor
        Final brightness temperature map [B, H, W].
    """
    device = config.device
    prefactor = config.prefactor

    # Compute brightness temperature δT_b
    bt_model = BrightnessTemperature(
        ionizationfraction=config.ionizationfraction,
        z=config.z,
        TCMB=config.TCMB,
        TS=config.TS,
        omegab=config.omegab,
        h=config.h,
        omegam=config.omegam,
        yp=config.yp
    )
    brightness = bt_model(delta)

    # Handle Gaussian image
    if gaussian_image is None:
        gaussian_image = torch.zeros_like(brightness)
    else:
        # Ensure Gaussian image has batch dimension
        if gaussian_image.dim() == 2:
            gaussian_image = gaussian_image.unsqueeze(0).expand_as(brightness)
        else:
            gaussian_image = gaussian_image.to(device)

    # Combine
    final_map = brightness + prefactor * gaussian_image
    return final_map









