import torch
import functionlistnew as func # Assuming your classes are here
import numpy as np
from pathlib import Path

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters
batch_size = 1
n_ant = 11
rank = n_ant

config = func.MatrixConfig(n_ant, rank, device)

# 1. Initialize Generators
S_gen = func.HollowIndefiniteMatrixGenerator(config)
G_gen = func.ThetaToDiagonalMatrix(config)
V_gen = func.GtoV(config)
Row_gen = func.VtoRow(config)

# 2. Generate S with the CORRECT batch size
# This is important! S must match the batch size of G.
S = S_gen(batch_size=batch_size)

# 3. Create dummy input z
# For n_ant=11, we need 2*(11-1) = 20 elements per batch
z = torch.rand(batch_size, 2*(n_ant-1))

# 4. Execution Pipeline
G = G_gen(z)           # Shape: (batch_size, n_ant, n_ant)
V = V_gen(G, S)        # Shape: (batch_size, n_ant, n_ant)
first_row = Row_gen(V) # Shape: (batch_size, n_ant)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# np.save(DATA_DIR / f'observed{n_ant}row.npy', first_row[0].detach().cpu().numpy())
np.save(DATA_DIR / f'observed{n_ant}V.npy', V[0].detach().cpu().numpy())
np.save(DATA_DIR / f'observed{n_ant}S.npy', S[0].unsqueeze(0).detach().cpu().numpy())
np.save(DATA_DIR / f'observed{n_ant}z.npy', z[0].detach().cpu().numpy())
