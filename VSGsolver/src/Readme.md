# VSGsolver - Visibility Gain Solver via SBI

Simulation-Based Inference pipeline for recovering complex antenna gains **G** from visibility matrices **V**, using the [Falcon](https://github.com/) framework with SNPE-A.

## Problem

Given a measured visibility matrix **V = G S G^H** (with zero diagonal), recover the complex gain matrix **G** where:
- **S** is a hollow indefinite Hermitian sky matrix (rank = n_ant)
- **G** is diagonal with g_1 = 1 + 0j (reference antenna)
- Parameters: 2*(n_ant - 1) real values encoding the remaining complex gains

## Directory Structure

```
VSGsolver/
├── config.yaml              # Falcon run configuration (priors, estimator, paths)
├── src/
│   ├── model.py             # Forward model: theta -> G -> V (simulator classes)
│   ├── functionlistnew.py   # Core math: MatrixConfig, S generator, G/V constructors
│   ├── embeddings.py        # Neural network embeddings for V (summary statistics)
│   ├── modelgen.ipynb       # Generate observed data (.npy files)
│   └── posterior.ipynb      # Visualize and analyze posterior samples
├── data/
│   ├── observed11V.npy      # Observed V matrix (n_ant x n_ant complex)
│   ├── observed11row.npy    # Observed first row of V
│   ├── observed11z.npy      # True gain parameters (for validation)
│   └── observed{N}*.npy     # Other antenna configurations (10, 40)
└── outputs/                  # Run outputs (gitignored)
    ├── 11antSingularValueEmbedding/
    ├── 11antallrws/
    ├── 11antbispectrum/
    ├── 11antclosurephase/
    ├── 11antrowmagnitude/
    ├── 11antfancy/
    ├── 10antennae*/
    └── 40antennae/
```

## Available Embeddings

Configured in `config.yaml` under `estimator.network.embedding`:

| Embedding | Class | Input | Output dim | Description |
|-----------|-------|-------|------------|-------------|
| Singular values | `SingularValueEmbedding` | V | n_ant | SVD of V matrix |
| All rows | `AllRowsEmbedding` | V | 2*n_ant^2 | Full V flattened to real |
| Bispectrum | `BispectrumEmbedding` | V | 2*C(n,3) | Triple products V_ij V_jk V_ki |
| Closure phases | `ClosurePhasesEmbedding` | V | C(n,3) | arg(V_ij V_jk conj(V_ik)) |
| Row magnitudes | `RowMagnitudeRatiosEmbedding` | V | n_ant | Row power ratios |
| First row | `FirstRowEmbedding` | V | 2*n_ant | First row of V |
| Log amplitudes | `LogAmplitudeEmbedding` | V | n_ant^2 | log\|V_ij\| |
| Selected rows | `SelectedRowsEmbedding` | V | 2*ceil(n/stride)*n | Every k-th row |

## How to Run a New Simulation

### 1. Configure

Edit `config.yaml`:
- Set priors count to `2*(n_ant - 1)` uniform [-1, 1] entries
- Set `logging.wandb.group` to a descriptive name (e.g., `11VGSsolverSingularValueEmbedding`)
- Choose embedding under `estimator.network.embedding._target_`
- Set `V.observed` to the correct `.npy` path in `data/`

### 2. Generate observations

Run `src/modelgen.ipynb` to create the observed `.npy` files. Set `n_ant` in `src/model.py` to match. Read the correct `S` file in `src/model.py`

### 3. Launch training

```bash
falcon launch config.yaml
```

### 4. Generate posterior samples

```bash
falcon samples config.yaml
```

### 5. Analyze results

Run `src/posterior.ipynb` to plot posteriors and compare against true parameters.

## .gitignore

From `03_gaincal` and `VSGsolver`, the following are tracked:
- `data/` and `src/` - all files regardless of type (including `.pth` model weights)
- `*.yaml`, `*.ipynb`, `*.md` - config, notebooks, and docs at each project root

Everything else is excluded:
- `outputs/` - Falcon run artifacts (wandb logs, graphs, samples)
- `*.joblib` - serialized training buffer snapshots
- `*.out` - SLURM job output logs
- `*.sh` - job submission scripts
- `__pycache__/` - Python bytecode
- `antennaecases/` - old antenna case studies

## Key Config Parameters

| Parameter | Current Value | Notes |
|-----------|--------------|-------|
| n_ant | 11 | Number of antennas |
| Priors | 20 x Uniform(-1, 1) | 2*(11-1) gain parameters |
| Embedding | SingularValueEmbedding | Via V matrix |
| Network | MAF | Masked Autoregressive Flow |
| Epochs | 100 | With early stopping (patience 32) |
| Buffer | 8000-10000 samples | Resampling every 5 rounds |
| Learning rate | 0.001 | With 0.5 decay, patience 16 |
