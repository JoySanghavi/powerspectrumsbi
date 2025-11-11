import torch
torch.manual_seed(42)
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Simulation Configuration
# ===============================
class SimulationConfig:
    """
    Centralized configuration for all constants and parameters.
    """
    def __init__(
        self,
        shape=128,
        BoxSize=128.0,
        sigma=10.0,
        prefactor=27.0,
        ionizationfraction=1.0,
        z=9.0,
        TCMB=0.0,
        TS=1.0,
        omegab=0.044,
        h=0.7,
        omegam=0.27,
        yp=0.248,
        nant=64,
        padded=None,
        subsetindex=None,
        device=None,
    ):
        # Grid
        self.shape = shape
        self.BoxSize = BoxSize
        self.dx = BoxSize / shape

        # Gaussian noise
        self.sigma = sigma
        self.prefactor = prefactor

        # Cosmology
        self.ionizationfraction = ionizationfraction
        self.z = z
        self.TCMB = TCMB
        self.TS = TS
        self.omegab = omegab
        self.h = h
        self.omegam = omegam
        self.yp = yp

        # Antenna/UV
        self.nant = nant
        self.padded = padded
        self.subsetindex = subsetindex

        # Device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# Baseline gain computation
# ===============================
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
    nant = gainlists.shape[1] // 2
    # Convert pairs of real/imag parts to complex numbers → shape: (B, N)
    gain_complex = torch.view_as_complex(gainlists.view(batch_size, nant, 2))
    # Compute all unique antenna index pairs (upper triangular, excluding diagonal)
    idx_i, idx_j = torch.triu_indices(nant, nant, offset=1)
    # Compute baseline gains: G_i * conj(G_j), shape → (B, P)
    baselinegains = gain_complex[:, idx_i] * torch.conj(gain_complex[:, idx_j])
    return baselinegains


# ===============================
# Theta to gains conversion
# ===============================
def thetatogains(theta: torch.Tensor):
    """
    Convert a tensor of theta values into a gains tensor by prepending (1, 0) 
    and incrementing all real parts by 1.
    Args:
        theta (torch.Tensor): A 2D tensor of shape (batch_size, 2 * (n-1)), where the real and imaginary parts
                            alternate along the last dimension. n generally is the
                            number of antennae in this case

    Returns:
        gains (torch.Tensor): A 2D tensor of shape (batch_size, 2 * (n)), where the first
                            pair is always (1, 0), followed by the modified theta values.

    Example:
        >>> theta = torch.tensor([[0.5, 0.2, 0.7, 0.3]])
        >>> thetatogains(theta)
        tensor([[1.0000, 0.0000, 1.5000, 0.2000, 1.7000, 0.3000]])
    """
    theta = theta.clone()
    theta[:, 0::2] += 1  # Add 1 to all real parts
    batch_size = theta.size(0)
    prepend_pair = torch.zeros((batch_size, 2), dtype=theta.dtype, device=theta.device)
    prepend_pair[:, 0] = 1.0
    gains = torch.cat((prepend_pair, theta), dim=1)
    return gains


# ===============================
# Multiply UV maps with Fourier image
# ===============================
def multiply_uvmaps_with_fourier(stacked: torch.Tensor, fourimag: torch.Tensor):
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
        fourimag = fourimag.unsqueeze(1)
    else:
        print(f"Fourier image dimension is {fourimag.dim()}. Incorrect results.")
    return stacked * fourimag


# ===============================
# UV Map Generator
# ===============================
class subsetUVMapGenerator:
    """
    subset UV Map Generator for baseline-weighted time accumulation.
    Precomputes fixed geometry.
    """
    def __init__(self, paddedtensor: torch.Tensor, subsetindices: torch.Tensor, shape: int):
        """
        Initialize the UV Map Generator.

        Parameters
        ----------
        paddedtensor : torch.Tensor
            Tensor of shape (N, 3, L) representing baseline positions and times.
        subsetindices : torch.Tensor
            Indices selecting a subset of baselines.
        shape : int
            Height and width of the UV map.
        """
        self.device = paddedtensor.device
        self.shape = shape
        self.subsetindices = subsetindices
        # Precompute selected subset → shape: (B, 3, L)
        self.selected = paddedtensor[subsetindices]
        # Extract x, y, t → shapes: (B, L)
        self.x = self.selected[:, 0, :].to(self.device)
        self.y = self.selected[:, 1, :].to(self.device)
        self.t = self.selected[:, 2, :].to(self.device)
        # Precompute flattened spatial indices → shape: (B, L)
        self.flat_idx_template = (self.y * self.shape + self.x).long()

    def __call__(self, baselinegain: torch.Tensor, mode = "norm") -> torch.Tensor:
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
        # Ensure baselinegain is on the same device
        baselinegain = baselinegain.to(self.device)
        Batchsize = baselinegain.shape[0]
        # Select relevant baseline gains → shape: (Batchsize, B)
        bg = baselinegain[:, self.subsetindices]  # select subset
        if mode =='norm':
            # Weighted time values → (Batchsize, B, L)
            weighted_t = torch.einsum('bi,il->bil', bg, self.t.to(baselinegain.dtype))
        elif mode =='square':
            # Weighted time values → (Batchsize, B, L)
            weighted_t = torch.einsum('bi,il->bil', (bg * bg.conj()), self.t.to(baselinegain.dtype))
        # Expand flat indices for batch → (Batchsize, B, L)
        flat_idx = self.flat_idx_template.unsqueeze(0).expand(Batchsize, -1, -1)
        # Reshape for scatter → (Batchsize, B*L)
        flat_idx = flat_idx.reshape(Batchsize, -1)
        weighted_t = weighted_t.reshape(Batchsize, -1)
        # Preallocate output on GPU
        output = torch.zeros(Batchsize, self.shape * self.shape, device=self.device, dtype=baselinegain.dtype)
        # Use scatter_add_, which is CUDA-optimized
        output.scatter_add_(1, flat_idx, weighted_t)
        # Reshape and fix orientation for visualization
        return output.view(Batchsize, self.shape, self.shape).transpose(1, 2)


class UVMapGenerator:
    """
    UVMapGenerator: Precomputes and stacks UV maps using subsetUVMapGenerator.
    Designed to return UV maps only, so that fourimag multiplication
    can be done outside.
    """
    def __init__(self, config: SimulationConfig):
        """
        Initialize the sampler.

        Parameters
        ----------
        config : Configuration file with paddedtensor positions, subsetindices and shape
        """
        self.config = config
        self.uvgens = [subsetUVMapGenerator(config.padded, subset, config.shape)
                       for subset in config.subsetindex]

    def __call__(self, gainlist: torch.Tensor, mode = "norm") -> torch.Tensor:
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
        baselinegain = baselinefromgain(gainlist)
        subuvmaps = [uvgen(baselinegain, mode).unsqueeze(1) for uvgen in self.uvgens]
        stacked = torch.cat(subuvmaps, dim=1)
        return stacked


class UnitUVMapGenerator:
    """
    Precompute UV maps for unit antenna gains.
    """
    def __init__(self, config: SimulationConfig):
        """
        Initialize and precompute stacked UV maps.

        Parameters
        ----------
        config: Configuraiton file which has the paddedtensor positions, subset of antennaes and shape
        and antennae number
        """
        self.config = config
        unittheta = torch.zeros((1, 2 * (config.nant-1)), device=config.device)
        unitgains = thetatogains(unittheta)
        uvgen = UVMapGenerator(config)
        self.stacked = uvgen(unitgains)

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
# Gaussian 2D Kernel
# ===============================
def gaussian_2d(size=128, sigma=10.0, device='cpu'):
    """
    Generate a 2D Gaussian kernel using PyTorch.

    Parameters
    ----------
    size : int, optional
        Size of the Gaussian kernel (size x size). Default: 128.
    sigma : float, optional
        Standard deviation of the Gaussian. Controls the spread. Default: 10.0.

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
    coords = torch.linspace(-(size - 1) / 2, (size - 1) / 2, steps=size, device=device)
    x, y = torch.meshgrid(coords, coords, indexing='xy')
    return torch.exp(-(x**2 + y**2) / (2 * sigma**2))


# ===============================
# Density Field and Power Spectrum
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
    import torch
    torch.manual_seed(42)
    real = torch.randn((B, H, W//2+1), device=device)
    imag = torch.randn((B, H, W//2+1), device=device)
    imag[..., 0] = 0.0
    if W % 2 == 0:
        imag[..., -1] = 0.0
    # torch.save(real, 'real.pth')
    # print(real)
    # print(imag)
    return real + 1j * imag


class DensityField2D:
    """
    Generate 2D Gaussian random density fields.
    """
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.shape = config.shape
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
        k = make_kgrid((self.shape, self.shape), self.BoxSize, self.device, rfft=True)
        pk = power_spectrum_2d(k, A, n) * self.dx**2
        noise = generate_hermitian_noise(B, self.shape, self.shape, self.device) * torch.sqrt(pk / 2)
        noise[..., 0, 0] = 0.0
        delta = torch.fft.irfft2(noise, s=(self.shape, self.shape), norm='ortho')
        return delta


# ===============================
# Brightness Temperature
# ===============================
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
    def __init__(self, config: SimulationConfig):
        self.config = config

    def __call__(self, density_field):
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
        cfg = self.config
        Tb = (
            27.0
            * cfg.ionizationfraction
            * (1 + density_field)
            * (((1 + cfg.z) / 10) ** 0.5)
            * (1 - cfg.TCMB / cfg.TS)
            * (cfg.omegab / 0.044)
            * (cfg.h / 0.7)
            * ((cfg.omegam / 0.27) ** -0.5)
            * ((1 - cfg.yp) / (1 - 0.248))
        )
        return Tb

def split_indices(n: int, k: int, device=None):
    """
    Splits indices [0, 1, ..., n-1] into k nearly equal parts.

    Args:
        n (int): Total number of elements.
        k (int): Number of splits.
        device (torch.device or str, optional): Device for returned tensors.

    Returns:
        List[torch.Tensor]: List of k tensors containing index splits.
    """
    if k <= 0:
        raise ValueError("Number of splits k must be > 0")
    if k > n:
        # Some splits will be empty, warn user
        print(f"Warning: Splitting {n} elements into {k} parts, some will be empty.")

    # Generate indices on the correct device
    device = device or torch.device("cpu")
    indices = torch.arange(n, device=device)

    # Calculate sizes: distribute remainder evenly
    base_size = n // k
    remainder = n % k
    sizes = [base_size + 1 if i < remainder else base_size for i in range(k)]

    # Use torch.split for efficient splitting
    return list(torch.split(indices, sizes))


# # ===============================
# # Full Simulation
# # ===============================
# class BrightnessTemperatureSimulation:
#     def __init__(self, config: SimulationConfig):
#         self.config = config
#         self.device = config.device
#         self.simulator = DensityField2D(config)
#         self.bt_model = BrightnessTemperature(config)
#         self.uv_generator = UVMapGenerator(config)
#         self.unit_uv = UnitUVMapGenerator(config)

#     def run_simulation(self, samples, A, n, plot=True):
#         samples = samples.to(self.device)
#         gains = thetatogains(samples)
#         stacked_uv = self.uv_generator(gains)

#         delta = self.simulator.sample_field(A, n).to(self.device)
#         Tb = self.bt_model(delta)

#         gaussianimage = gaussian_2d(size=self.config.shape, sigma=self.config.sigma, device=self.device)
#         gaussianimage = gaussianimage.unsqueeze(0).expand_as(Tb)
#         final_map = Tb + self.config.prefactor * gaussianimage

#         finaluvmap = torch.fft.fft2(final_map, norm="ortho")
#         tracked_uv = multiply_uvmaps_with_fourier(stacked_uv, finaluvmap)
#         normed_tracked_uv = tracked_uv / self.unit_uv()
#         normed_tracked_uv = normed_tracked_uv.nan_to_num(nan=0.0)

#         if plot:
#             plt.figure()
#             plt.imshow(final_map[0].detach().cpu())
#             plt.colorbar()
#             plt.title("Final Brightness Map")

#             plt.figure()
#             plt.imshow(torch.real(torch.fft.ifft2(normed_tracked_uv[0, 0], norm="ortho")).detach().cpu())
#             plt.colorbar()
#             plt.title("Reconstructed Map")

#         return {
#             "A": A,
#             "n": n,
#             "delta": delta,
#             "brightness_temperature": Tb,
#             "final_map": final_map,
#             "finaluvmap": finaluvmap,
#             "tracked_uv": tracked_uv,
#             "normed_tracked_uv": normed_tracked_uv,
#         }



# # ===============================
# # Example Usage
# # ===============================



# padded = torch.load("paddedAA2_4_10_9_128.pth")
# indices1 = torch.arange(1008)
# indices2 = torch.arange(1008, 2016)
# subsetindex = [indices1, indices2]
# # # Example 1: Split 10 indices into 3 parts
# # splits = split_indices(20, 4)
# # for i, s in enumerate(splits):
# #     print(f"Split {i+1}:", s)

# config = SimulationConfig(
#     shape=128,
#     BoxSize=128.0,
#     nant=64,
#     padded=padded,
#     subsetindex=subsetindex
# )

# sim = BrightnessTemperatureSimulation(config)

# samples = torch.randn(3, 2*(config.nant-1))*0.1  # theta shape
# torch.manual_seed(54)
# An = torch.abs(torch.randn(3,2))
# A = An[:,0]/100
# n = -An[:,1]*5

# results = sim.run_simulation(samples, A=A, n=n, plot=True)
