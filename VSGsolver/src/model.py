import torch
import functionlistnew as func
import falcon
import wandb
import numpy as np
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
n_ant=11
rank=n_ant
config = func.MatrixConfig(n_ant, rank, device)

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
padded_path = DATA_DIR / "observed11S.npy"

S = torch.tensor(np.load(padded_path)).to(device)
Gcreator = func.ThetaToDiagonalMatrix(config)
Vcreator = func.GtoV(config)
Rowcreator = func.VtoRow(config)

class creategainmatrix:
    """Gain Matrix created based on parameters theta/z."""
    def simulate_batch(self, batch_size, z):
        z = torch.tensor(z)
        G = Gcreator(z)
        return G.detach().cpu().numpy()

class createvfromg:
    """V Matrix created based on gains."""
    def simulate_batch(self, batch_size, G):
        G = torch.tensor(G, device=device)
        V = Vcreator(G, S)
        return V.detach().cpu().numpy()

class create1strowfromv:
    """Read 1st row of V"""
    def simulate_batch(self, batch_size, V):
        V = torch.tensor(V, device=device)
        row = Rowcreator(V)
        return row.detach().cpu().numpy()

import torch.nn as nn

class ComplexFlatten(nn.Module):
    """
    A PyTorch Module that converts a complex tensor into a real-valued 
    concatenated tensor (Real | Imaginary).
    """
    def __init__(self):
        super(ComplexFlatten, self).__init__()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Complex tensor of shape (Batch, N)
        Returns:
            torch.Tensor: Real tensor of shape (Batch, 2*N)
        """
        if not torch.is_complex(x):
            # Using a warning or error ensures your pipeline doesn't 
            # silently fail with garbage data.
            raise TypeError(f"Expected complex tensor, but got {x.dtype}")

        # Extract components
        real = x.real
        imag = x.imag

        # Concatenate along the last dimension
        # (Batch, N) + (Batch, N) -> (Batch, 2*N)
        return torch.cat([real, imag], dim=-1)

# Usage in your pipeline:
# flattener = ComplexFlatten()
# output = flattener(row)
