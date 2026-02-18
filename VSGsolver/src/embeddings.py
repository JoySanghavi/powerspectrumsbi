import torch
import torch.nn as nn


class ComplexToReal(nn.Module):
    """
    Helper module to convert complex tensor to real by concatenating
    real and imaginary parts along the last dimension.
    """
    def __init__(self):
        super(ComplexToReal, self).__init__()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Complex tensor of shape (..., N)
        Returns:
            torch.Tensor: Real tensor of shape (..., 2*N)
        """
        if not torch.is_complex(x):
            return x
        return torch.cat([x.real, x.imag], dim=-1)


class BispectrumEmbedding(nn.Module):
    """
    Computes bispectrum (triple products) for antenna triplets.
    For triangles (i,j,k): V[i,j] * V[j,k] * V[k,i]
    This is proportional to |g_i|² |g_j|² |g_k|² * (S_ij * S_jk * S_ki)

    Output: Complex bispectra flattened to real (real | imag)
    """
    def __init__(self, max_triplets=None):
        """
        Args:
            max_triplets: Maximum number of triplets to compute (None = all)
        """
        super(BispectrumEmbedding, self).__init__()
        self.max_triplets = max_triplets
        self.complex_to_real = ComplexToReal()
        self._triplets_cache = None

    def _get_triplets(self, n_ant):
        """Generate triplet indices for given number of antennas."""
        triplets = []
        for i in range(n_ant):
            for j in range(i+1, n_ant):
                for k in range(j+1, n_ant):
                    triplets.append((i, j, k))
                    if self.max_triplets and len(triplets) >= self.max_triplets:
                        return triplets
        return triplets

    def forward(self, V):
        """
        Args:
            V (torch.Tensor): Complex visibility matrix (Batch, n_ant, n_ant)
        Returns:
            torch.Tensor: Real tensor (Batch, 2 * n_triplets)
        """
        n_ant = V.shape[1]

        # Cache triplets for efficiency
        if self._triplets_cache is None or len(self._triplets_cache) != n_ant:
            self._triplets_cache = self._get_triplets(n_ant)

        triplets = self._triplets_cache
        bispectra = []

        for i, j, k in triplets:
            # V[i,j] * V[j,k] * V[k,i]
            bispectrum = V[:, i, j] * V[:, j, k] * V[:, k, i]
            bispectra.append(bispectrum)

        bispectra = torch.stack(bispectra, dim=-1)  # (Batch, n_triplets)
        return self.complex_to_real(bispectra)


class AllRowsEmbedding(nn.Module):
    """
    Flattens all rows of V matrix.
    Each row i: V[i,:] = g_i * S[i,:] * conj(g)

    Output: All matrix elements flattened to real vector
    """
    def __init__(self):
        super(AllRowsEmbedding, self).__init__()
        self.complex_to_real = ComplexToReal()

    def forward(self, V):
        """
        Args:
            V (torch.Tensor): Complex visibility matrix (Batch, n_ant, n_ant)
        Returns:
            torch.Tensor: Real tensor (Batch, 2 * n_ant * n_ant)
        """
        batch_size = V.shape[0]
        V_flat = V.reshape(batch_size, -1)  # (Batch, n_ant * n_ant)
        return self.complex_to_real(V_flat)


class RowMagnitudeRatiosEmbedding(nn.Module):
    """
    Computes row magnitude ratios relative to first antenna.
    For row i: sum_j |V[i,j]|² / sum_j |V[0,j]|²
    This gives information about |g_i|² / |g_0|² ≈ |g_i|² (since g_0=1)

    Output: Real-valued ratios
    """
    def __init__(self, eps=1e-10):
        super(RowMagnitudeRatiosEmbedding, self).__init__()
        self.eps = eps

    def forward(self, V):
        """
        Args:
            V (torch.Tensor): Complex visibility matrix (Batch, n_ant, n_ant)
        Returns:
            torch.Tensor: Real tensor (Batch, n_ant)
        """
        row_mags_sq = torch.abs(V) ** 2  # (Batch, n_ant, n_ant)
        row_sums = row_mags_sq.sum(dim=-1)  # (Batch, n_ant)

        # Normalize by first antenna
        ratios = row_sums / (row_sums[:, 0:1] + self.eps)
        return ratios


class ClosurePhasesEmbedding(nn.Module):
    """
    Computes closure phases for antenna triplets.
    For triangles (i,j,k): arg(V[i,j] * V[j,k] * conj(V[i,k]))
    These are gain-phase independent and help constrain S.

    Output: Real-valued phases in radians
    """
    def __init__(self, max_triplets=None):
        super(ClosurePhasesEmbedding, self).__init__()
        self.max_triplets = max_triplets
        self._triplets_cache = None

    def _get_triplets(self, n_ant):
        """Generate triplet indices for given number of antennas."""
        triplets = []
        for i in range(n_ant):
            for j in range(i+1, n_ant):
                for k in range(j+1, n_ant):
                    triplets.append((i, j, k))
                    if self.max_triplets and len(triplets) >= self.max_triplets:
                        return triplets
        return triplets

    def forward(self, V):
        """
        Args:
            V (torch.Tensor): Complex visibility matrix (Batch, n_ant, n_ant)
        Returns:
            torch.Tensor: Real tensor (Batch, n_triplets)
        """
        n_ant = V.shape[1]

        # Cache triplets for efficiency
        if self._triplets_cache is None or len(self._triplets_cache) != n_ant:
            self._triplets_cache = self._get_triplets(n_ant)

        triplets = self._triplets_cache
        closures = []

        for i, j, k in triplets:
            # arg(V[i,j] * V[j,k] * conj(V[i,k]))
            closure_phase = torch.angle(V[:, i, j] * V[:, j, k] * torch.conj(V[:, i, k]))
            closures.append(closure_phase)

        return torch.stack(closures, dim=-1)


class SingularValueEmbedding(nn.Module):
    """
    Computes singular values of V matrix.
    Captures global amplitude structure.

    Output: Real-valued singular values
    """
    def __init__(self, n_singular_values=None):
        """
        Args:
            n_singular_values: Number of top singular values to keep (None = all)
        """
        super(SingularValueEmbedding, self).__init__()
        self.n_singular_values = n_singular_values

    def forward(self, V):
        """
        Args:
            V (torch.Tensor): Complex visibility matrix (Batch, n_ant, n_ant)
        Returns:
            torch.Tensor: Real tensor (Batch, n_singular_values)
        """
        batch_size = V.shape[0]
        singular_values = []

        for b in range(batch_size):
            _, S_vals, _ = torch.linalg.svd(V[b])
            if self.n_singular_values is not None:
                S_vals = S_vals[:self.n_singular_values]
            singular_values.append(S_vals)

        return torch.stack(singular_values)


class LogAmplitudeEmbedding(nn.Module):
    """
    Computes log-amplitudes of V matrix elements.
    log|V[i,j]| = log|g_i| + log|g_j| + log|S[i,j]|
    This linearizes the gain problem!

    Output: Real-valued log amplitudes
    """
    def __init__(self, eps=1e-10):
        super(LogAmplitudeEmbedding, self).__init__()
        self.eps = eps

    def forward(self, V):
        """
        Args:
            V (torch.Tensor): Complex visibility matrix (Batch, n_ant, n_ant)
        Returns:
            torch.Tensor: Real tensor (Batch, n_ant * n_ant)
        """
        batch_size = V.shape[0]
        log_amps = torch.log(torch.abs(V) + self.eps)
        return log_amps.reshape(batch_size, -1)


class SelectedRowsEmbedding(nn.Module):
    """
    Selects specific rows of V (e.g., every k-th row, or specific antennas).
    Useful for reducing dimensionality while preserving information.

    Output: Selected rows flattened to real vector
    """
    def __init__(self, stride=5):
        """
        Args:
            stride: Take every stride-th row
        """
        super(SelectedRowsEmbedding, self).__init__()
        self.stride = stride
        self.complex_to_real = ComplexToReal()

    def forward(self, V):
        """
        Args:
            V (torch.Tensor): Complex visibility matrix (Batch, n_ant, n_ant)
        Returns:
            torch.Tensor: Real tensor (Batch, 2 * len(row_indices) * n_ant)
        """
        batch_size = V.shape[0]
        n_ant = V.shape[1]
        row_indices = list(range(0, n_ant, self.stride))
        selected = V[:, row_indices, :]  # (Batch, len(row_indices), n_ant)
        selected_flat = selected.reshape(batch_size, -1)
        return self.complex_to_real(selected_flat)


class FirstRowEmbedding(nn.Module):
    """
    Extracts first row of V (already implemented in model.py as VtoRow).
    Included here for completeness.

    Output: First row flattened to real vector
    """
    def __init__(self):
        super(FirstRowEmbedding, self).__init__()
        self.complex_to_real = ComplexToReal()

    def forward(self, V):
        """
        Args:
            V (torch.Tensor): Complex visibility matrix (Batch, n_ant, n_ant)
        Returns:
            torch.Tensor: Real tensor (Batch, 2 * n_ant)
        """
        row = V[:, 0, :]  # (Batch, n_ant)
        return self.complex_to_real(row)


class CombinedEmbedding(nn.Module):
    """
    Combines multiple embeddings into a single output vector.
    Useful for creating rich feature representations.
    """
    def __init__(self, embeddings):
        """
        Args:
            embeddings: List of embedding modules
        """
        super(CombinedEmbedding, self).__init__()
        self.embeddings = nn.ModuleList(embeddings)

    def forward(self, V):
        """
        Args:
            V (torch.Tensor): Complex visibility matrix (Batch, n_ant, n_ant)
        Returns:
            torch.Tensor: Real tensor (Batch, total_embedding_dim)
        """
        outputs = [emb(V) for emb in self.embeddings]
        return torch.cat(outputs, dim=-1)


# Example usage and factory functions
def create_standard_embedding(device='cpu'):
    """
    Creates a standard combined embedding with commonly used statistics.

    Includes:
    - Phase info: First row (complex)
    - Magnitude constraints: Row ratios, bispectrum, closure phases, SVD, log-amps

    Returns:
        CombinedEmbedding module
    """
    embeddings = [
        FirstRowEmbedding(),                    # Phase info ✓
        RowMagnitudeRatiosEmbedding(),          # Magnitude only
        BispectrumEmbedding(max_triplets=100),  # Magnitude (G²) constraint
        ClosurePhasesEmbedding(max_triplets=100),  # Sky constraint
        SingularValueEmbedding(n_singular_values=10),  # Global magnitude
        LogAmplitudeEmbedding(),                # Linearized magnitude
    ]
    return CombinedEmbedding(embeddings).to(device)


def create_phase_focused_embedding(stride=2, max_triplets=50, device='cpu'):
    """
    Emphasizes phase information with selected rows + magnitude constraints.
    Good balance between phase recovery and computational efficiency.

    Includes:
    - Phase info: Multiple rows (every stride-th row)
    - Magnitude constraints: Bispectrum, row ratios

    Args:
        stride: Take every stride-th row (stride=2 means every other row)
        max_triplets: Max bispectrum terms

    Returns:
        CombinedEmbedding module
    """
    embeddings = [
        SelectedRowsEmbedding(stride=stride),     # Phase info from multiple rows ✓
        BispectrumEmbedding(max_triplets=max_triplets),  # G² magnitude constraint
        RowMagnitudeRatiosEmbedding(),            # Direct |g_i|² ratios
    ]
    return CombinedEmbedding(embeddings).to(device)


def create_full_information_embedding(max_triplets=100, device='cpu'):
    """
    Maximum information: full V matrix + all magnitude constraints.
    Best for complex G recovery, but high dimensional.

    Includes:
    - Phase info: All rows of V (complete phase information)
    - Magnitude constraints: Bispectrum, row ratios, singular values

    Args:
        max_triplets: Max bispectrum/closure terms

    Returns:
        CombinedEmbedding module
    """
    embeddings = [
        AllRowsEmbedding(),                       # Full phase info ✓✓✓
        BispectrumEmbedding(max_triplets=max_triplets),  # G² constraint
        RowMagnitudeRatiosEmbedding(),            # |g_i|² ratios
        SingularValueEmbedding(n_singular_values=10),  # Global structure
    ]
    return CombinedEmbedding(embeddings).to(device)


def create_minimal_phase_embedding(device='cpu'):
    """
    Minimal embedding: just first row with bispectrum constraint.
    Lightest option that still recovers full complex G.

    Includes:
    - Phase info: First row only
    - Magnitude constraint: Bispectrum (G²)

    Returns:
        CombinedEmbedding module
    """
    embeddings = [
        FirstRowEmbedding(),                      # Phase info ✓
        BispectrumEmbedding(max_triplets=50),     # G² magnitude constraint
    ]
    return CombinedEmbedding(embeddings).to(device)
