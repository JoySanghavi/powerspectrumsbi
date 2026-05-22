import numpy as np
import torch
import torch.nn as nn
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
import falcon
import wandb
import functionlistnew as func
from pathlib import Path

# Global configuration
HERE = Path(__file__).resolve().parent
padded_path = HERE / "paddedAA2_4_10_9_128.pth"
padded = torch.load(padded_path)
subsetindex = func.split_indices(padded.shape[0], 2)
noise = 0.0
knownant = 59 #64-1-unknownantennae
config = func.SimulationConfig(
    shape=128,
    BoxSize=128.0,
    prefactor = 2.7,
    nant=64,
    padded=padded,
    subsetindex=subsetindex
)
PWSimulator = func.DensityField2D(config)
BT_model = func.BrightnessTemperature(config)
UV_generator = func.UVMapGenerator(config)
Unit_uv = func.UnitUVMapGenerator(config)

class creategainfromtheta:
    """Gains created based on parameters theta/z."""
    def simulate_batch(self, batch_size, z):
        z = torch.tensor(z, device=device)
        zeros = torch.zeros(z.size(0), int(knownant*2))
        # Concatenate along the feature dimension (dim=1)
        z = torch.cat([z, zeros], dim=1)
        gains = func.thetatogains(z)
        return gains.detach().cpu().numpy()

class createuvmapfromgain:
    """UVmaps created based on parameters gains."""
    def simulate_batch(self, batch_size, gains):
        gains = torch.tensor(gains, device=device)
        stacked_uv = UV_generator(gains)
        return stacked_uv.detach().cpu().numpy()

class createnoiseuvmapfromgain:
    """Noise UVmaps created based on parameters gains."""
    def simulate_batch(self, batch_size, gains):
        gains = torch.tensor(gains, device=device)
        noisestacked_uv = UV_generator(gains, mode='square')
        return noisestacked_uv.detach().cpu().numpy()

class createdffrompw:
    """Density field created based on parameters an."""
    def simulate_batch(self, batch_size, pw):
        pw = torch.tensor(pw, device=device)
        a = torch.pow(10, pw[:,0])
        n = pw[:,1]
        # a = torch.zeros(pw.size(0))
        # n = torch.zeros(pw.size(0))
        df = PWSimulator.sample_field(a, n)
        return df.detach().cpu().numpy()

class createbtfromdf:
    """Brightness temperature field created based on parameters densityfields."""
    def simulate_batch(self, batch_size, df):
        df = torch.tensor(df, device=device)
        bt = BT_model(df)
        return bt.detach().cpu().numpy()       

class createimagefrombtandgauss:
    """Image created based on parameters bt and gauss."""
    def simulate_batch(self, batch_size, bt):
        bt = torch.tensor(bt, device=device)
        foregroundimage = func.gaussian_2d(size=config.shape, sigma=config.sigma)
        foregroundimage = foregroundimage.unsqueeze(0).expand_as(bt)
        image = bt + config.prefactor * foregroundimage
        return image.detach().cpu().numpy()

class createuvimagefromimage:
    """UV Image created based on parameters image"""
    def simulate_batch(self, batch_size, image):
        image = torch.tensor(image, device=device)
        uvimage = torch.fft.fft2(image, norm="ortho")
        return uvimage.detach().cpu().numpy()

class Noise:
    """Gaussian noise generator."""
    def simulate_batch(self, batch_size, uvimage):
        uvimage = torch.tensor(uvimage, device=device)
        noiseimage = torch.rand_like(uvimage)*noise
        return noiseimage.detach().cpu().numpy()

class createsvfromuvimageandtracks:
    """Summed Visibilty created based on parameters uvimage and tracks"""
    def simulate_batch(self, batch_size, uvimage, stacked_uv):
        uvimage = torch.tensor(uvimage, device=device)
        stacked_uv = torch.tensor(stacked_uv, device=device)
        tracked_uv = func.multiply_uvmaps_with_fourier(stacked_uv, uvimage)
        return tracked_uv.detach().cpu().numpy()

class createnvfromsv:
    """Normalised Visibility created based on summed visibility"""
    def simulate_batch(self, batch_size, tracked_uv):
        tracked_uv = torch.tensor(tracked_uv, device=device)
        normed_tracked_uv = tracked_uv / Unit_uv()
        normed_tracked_uv = normed_tracked_uv.nan_to_num(nan=0.0)
        return normed_tracked_uv.detach().cpu().numpy()


class createsvfromnoiseandtracks:
    """Summed noise visibilty created based on parameters noiseimage and tracks"""
    def simulate_batch(self, batch_size, noiseimage, noisestacked_uv):
        noiseimage = torch.tensor(noiseimage, device=device)
        noisestacked_uv = torch.tensor(noisestacked_uv, device=device)
        noisestacked_uv = torch.sqrt(noisestacked_uv)
        noisetracked_uv = func.multiply_uvmaps_with_fourier(noisestacked_uv, noiseimage)
        return noisetracked_uv.detach().cpu().numpy()

class createnoisenvfromsv:
    """Normalised noise Visibility created based on summed noise visibility"""
    def simulate_batch(self, batch_size, noisetracked_uv):
        noisetracked_uv = torch.tensor(noisetracked_uv, device=device)
        noisenormed_tracked_uv = noisetracked_uv / Unit_uv()
        noisenormed_tracked_uv = noisenormed_tracked_uv.nan_to_num(nan=0.0)
        return noisenormed_tracked_uv.detach().cpu().numpy()

class createxfromnvandnoisenv:
    """x created based on noise and image normalized visibilities"""
    def simulate_batch(self, batch_size, normed_tracked_uv, noisenormed_tracked_uv):
        normed_tracked_uv = torch.tensor(normed_tracked_uv, device=device)
        noisenormed_tracked_uv = torch.tensor(noisenormed_tracked_uv, device=device)
        x = normed_tracked_uv + noisenormed_tracked_uv
        return x.detach().cpu().numpy()

class createobsxfromx:
    """x created based on noise and image normalized visibilities"""
    def simulate_batch(self, batch_size, x):
        x = torch.tensor(x)
        obsx = torch.fft.ifft2(x, norm = 'ortho')

        real = obsx.real      # (B,2,H,W)
        imag = obsx.imag      # (B,2,H,W)
        # Concatenate along channel dimension → (B,4,H,W)
        obsx = torch.cat([real, imag], dim=1)

        return obsx.detach().cpu().numpy()

import torch.nn.functional as F
import timm

class E(nn.Module):
    """
    Embedding network using a timm CNN encoder with per-slice layer normalization.
    """
    def __init__(self, backbone='resnet18', in_chans=4, log_prefix=None):
        super().__init__()
        self.log_prefix = log_prefix + ":" if log_prefix else ""

        # 1. Pretrained timm encoder (remove classifier)
        base = timm.create_model(backbone, pretrained=True, in_chans=in_chans)
        self.encoder = base

    def forward(self, x, *args):
        """
        x: (B, N, H, W) or (B, C, H, W)
        Returns: (B, latent_dim)
        """
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        if x.ndim == 4:
            B, N, H, W = x.shape
        else:
            raise ValueError("Expected x with shape (B, N, H, W)")

        x = x/27.0

        # Pass through pretrained encoder
        h = self.encoder(x)                     # (B, feature_dim, 1, 1)
        out = h.flatten(1)                        # (B, feature_dim)
        return out
# import torch.nn as nn
# import timm

# class E(nn.Module):
#     """
#     Efficient Embedding network using MobileNetV3-Small.
#     """
#     def __init__(self, backbone='resnet18', in_chans=4, embedding_dim=64, log_prefix=None):
#         super().__init__()
#         self.log_prefix = log_prefix + ":" if log_prefix else ""
#         # 1. Load Pretrained MobileNetV3
#         # num_classes=0 removes the final classifier.
#         # global_pool='' keeps the spatial features initially if we need to process them, 
#         # but here we let timm handle pooling by default or manually below.
#         self.encoder = timm.create_model(
#             backbone, 
#             pretrained=True, 
#             in_chans=in_chans, 
#             num_classes=0  # Returns the feature vector (B, 576)
#         )
        
#         # Optional: Further compression
#         # If 576 is still too big for your storage, you can project it down (e.g., to 128).
#         # Set embedding_dim to an integer (e.g., 128) to enable this.
#         self.projection = None
#         if embedding_dim:
#             # Dynamically get the output features of the encoder (usually 576 for mobilenetv3_small)
#             enc_features = self.encoder.num_features
#             self.projection = nn.Linear(enc_features, embedding_dim)

#     def forward(self, x, *args):
#         """
#         x: (B, C, H, W)
#         Returns: (B, 576) or (B, embedding_dim)
#         """
#         # Ensure input is a tensor and on correct device
#         if not torch.is_tensor(x):
#              x = torch.as_tensor(x, dtype=torch.float32)
#         if x.device != next(self.parameters()).device:
#             x = x.to(next(self.parameters()).device)

#         # Custom normalization (Your logic)
#         x = x / 27.0

#         # Pass through encoder
#         # Output is (B, 576)
#         features = self.encoder(x)

#         if features.ndim == 4:           # (B, C, H, W)
#             features = features.mean(dim=[2, 3])   # global avg pool → (B, C)

#         # Optional projection to smaller size
#         if self.projection:
#             features = self.projection(features)            
#         return features
# class E(torch.nn.Module):
#     """Embedding network flattening high-dimensional observations per slice with normalization."""
#     def __init__(self, log_prefix=None):
#         super(E, self).__init__()
#         self.log_prefix = log_prefix + ":" if log_prefix else ""

#     def forward(self, x, *args):
#         # falcon.log({f"{self.log_prefix}input_min": x.min().item()})
#         # falcon.log({f"{self.log_prefix}input_max": x.max().item()})
#         # x shape: (B, N, 128, 128)
#         x = torch.tensor(x, dtype=torch.float32, device=device)
#         B, N, H, W = x.shape
#         # Step 1: Flatten each 128x128 slice
#         x = x.view(B, N, H * W)  # (B, N, 16384)
#         # Step 2: Normalize each slice independently
#         x = torch.nn.functional.layer_norm(x, x.shape[-1:])  # (B, N, 16384)
#         # Step 3: Flatten N if needed to get one vector per batch
#         x = x.view(B, N * H * W)  # (B, N*16384)
#         # falcon.log({f"{self.log_prefix}output_min": x.min().item()})
#         # falcon.log({f"{self.log_prefix}output_max": x.max().item()})
#         # print("E(x) generated")
#         return x

# class E(nn.Module):
#     """Embedding network applying CNN to multi-channel observations (N like RGB) with per-channel normalization."""
#     def __init__(self, log_prefix=None, embedding_dim=3):
#         super(E, self).__init__()
#         self.log_prefix = log_prefix + ":" if log_prefix else ""

#         # CNN layers
#         self.conv = nn.Sequential(
#             nn.LazyConv2d(out_channels=32, kernel_size=3, stride=2, padding=1),  # Lazy in_channels=N
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1))  # (B, 128, 1, 1)
#         )

#         # Linear layer to final embedding
#         self.fc = nn.Linear(128, embedding_dim)

#     def forward(self, x, *args):
#         falcon.log({f"{self.log_prefix}input_min": x.min().item()})
#         falcon.log({f"{self.log_prefix}input_max": x.max().item()})

#         # falcon.log({f"{self.log_prefix}input_min": x.min().item()})
#         # falcon.log({f"{self.log_prefix}input_max": x.max().item()})
#         # x shape: (B, N, 128, 128)
#         B, N, H, W = x.shape
#         # Step 1: Flatten each 128x128 slice
#         x = x.view(B, N, H * W)  # (B, N, 16384)
#         # Step 2: Normalize each slice independently
#         x = nn.functional.layer_norm(x, x.shape[-1:])  # (B, N, 16384)
#         x = x.view(B, N, H, W)                         # (B, C, H, W)
#         # Apply CNN
#         x = self.conv(x)                        # (B, 128, 1, 1)
#         x = x.view(B, -1)                       # (B, 128)
#         x = self.fc(x)                          # (B, embedding_dim)
#         return x

# class E(nn.Module):
#     """Embedding network with online normalization.
#     Args:
#         momentum: Momentum for online normalization (default: 0.01)
#     """
#     def __init__(self, momentum: float = 1e-2):
#         super(E, self).__init__()
#         self.norm = falcon.contrib.LazyOnlineNorm(momentum=momentum)

#     def forward(self, x):
#         falcon.log({"norm_pre": x.std().item()})
#         x = self.norm(x).float()
#         falcon.log({"norm_post": x.std().item()})
#         return x