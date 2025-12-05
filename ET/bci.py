import torch
import torch.nn as nn
from typing import Optional

def construct_A_matrix(D: torch.Tensor, cutoff: Optional[float] = None):
    """
    Construct the A matrix using a distance cutoff on the distance matrix D.

    Parameters
    ----------
    D (N, N): Pairwise distance matrix (D[i, j] = distance between atoms i and j).
    cutoff  : Distance cutoff; any pair (i, j) with D[i, j] < cutoff is treated as a bond.
              If cutoff is None, construct a fully connected graph (all i < j).

    Returns
    -------
    A (N, M)       : Incidence matrix.
    bond_indices   : LongTensor of shape (M, 2) with bonded atom index pairs (i, j), i < j.
    """
    N = D.shape[0]
    device = D.device
    dtype = D.dtype

    with torch.no_grad():
        upper_mask = torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=1)

        if cutoff is None:
            # Fully connected graph: all off-diagonal upper-tri entries are bonds
            bond_mask = upper_mask
        else:
            # Use cutoff
            cutoff_mask = (D < cutoff)
            bond_mask = cutoff_mask & upper_mask

        bond_indices = bond_mask.nonzero(as_tuple=False)  # (M, 2)

    M = bond_indices.shape[0]

    if M == 0:
        A = torch.zeros((N, 0), dtype=dtype, device=device)
        return A, bond_indices  # empty

    i = bond_indices[:, 0]
    j = bond_indices[:, 1]
    e = torch.arange(M, device=device)

    A = torch.zeros((N, M), dtype=dtype, device=device)
    A[i, e] = 1.0
    A[j, e] = -1.0

    return A, bond_indices


class QEqBCI_penalty(QEq):
    """
    QEq implementation with bond charge increments (BCI).

    k   = A^T J A + 2 * P_diag
    bci = - k^-1 A^T (chi + J F)
    q   = F + bci
    """

    def __init__(
        self,
        incline_init: float = 10.0,
        displacement_init: float = 1.0,
        height_init: float = 1.0,
        h_cutoff_init: float = 5.0,
    ):
        super().__init__()

        # Learnable scalar parameters
        self.incline = nn.Parameter(torch.tensor(incline_init))
        self.displacement = nn.Parameter(torch.tensor(displacement_init))
        self.height = nn.Parameter(torch.tensor(height_init))

        # Non-learnable cutoff (buffer)
        self.register_buffer("h_cutoff", torch.tensor(h_cutoff_init))

        self._number_extra_features = 4 # Is this correct?

    def forward(
        self,
        net_charges: torch.Tensor,      # unused
        features: torch.Tensor,         # unused
        species: torch.Tensor,          # unused
        chi: torch.Tensor,
        jii: torch.Tensor,
        coordinates: torch.Tensor,
        F: torch.Tensor,                # formal charges
        interaction: int,               # unused
        num_species: int,               # unused
        rnn_count: int,                 # unused
        net_spins: Optional[torch.Tensor] = None,   # unused
        ext_field: Optional[torch.Tensor] = None,   # unused
    ):
        # Pairwise distances D
        norms = (coordinates ** 2).sum(dim=1, keepdim=True)      # (N, 1)
        D_squared = norms + norms.T - 2 * coordinates @ coordinates.T
        D = torch.sqrt(torch.clamp(D_squared, min=0.0))

        # Incidence matrix A using fixed cutoff
        A, bonds = construct_A_matrix(D, float(self.h_cutoff.item()))

        # If no bonds, just return F as charges
        if A.shape[1] == 0:
            pred_charges = F.clone()
            return pred_charges, coordinates.new_zeros(species.shape)

        # Bond vectors: coordinates is (N, 3) → (3, N) @ (N, M) = (3, M)
        bond_vecs = coordinates.T @ A
        bond_lengths = torch.linalg.norm(bond_vecs, dim=0)       # (M,)

        # Smooth, differentiable P with learnable parameters
        s = self.incline * (bond_lengths - self.displacement)    # (M,)
        P = torch.sigmoid(s) * self.height                       # (M,)

        # Build matrices for QEq BCI
        J = torch.diag(jii)                    # (N, N)
        P_diag = torch.diag(P)                 # (M, M)
        K = A.T @ J @ A + 2 * P_diag           # (M, M)
        rhs = - A.T @ (chi + J @ F)            # (M,)

        # Solve 
        b = torch.linalg.pinv(K) @ rhs         # (M,)
        pred_charges = F + A @ b               # (N,)

        return pred_charges, coordinates.new_zeros(species.shape)

class QEqBCI_smooth_A(QEq):
    """
    QEq implementation with bond charge increments (BCI) and
    learnable smooth A matrix.
    """

    def __init__(
        self,
        incline_init: float = 10.0,
        displacement_init: float = 1.0,
        height_init: float = 1.0,
    ):
        super().__init__()

        # Learnable scalar parameters
        self.incline = nn.Parameter(torch.tensor(incline_init))
        self.displacement = nn.Parameter(torch.tensor(displacement_init))
        self.height = nn.Parameter(torch.tensor(height_init))

        self._number_extra_features = 4

    def forward(
        self,
        net_charges: torch.Tensor,      # unused
        features: torch.Tensor,         # unused
        species: torch.Tensor,          # unused
        chi: torch.Tensor,
        jii: torch.Tensor,
        coordinates: torch.Tensor,
        F: torch.Tensor,                # formal charges
        interaction: int,               # unused
        num_species: int,               # unused
        rnn_count: int,                 # unused
        net_spins: Optional[torch.Tensor] = None,   # unused
        ext_field: Optional[torch.Tensor] = None,   # unused
    ):
        device = coordinates.device
        dtype = coordinates.dtype

        # Pairwise distances D
        norms = (coordinates ** 2).sum(dim=1, keepdim=True)      # (N, 1)
        D_squared = norms + norms.T - 2 * coordinates @ coordinates.T
        D = torch.sqrt(torch.clamp(D_squared, min=0.0))

        # Incidence matrix A using fixed cutoff
        A, bonds = construct_A_matrix(D, float(self.h_cutoff.item()))

        # If no bonds, just return F as charges
        if A.shape[1] == 0:
            pred_charges = F.clone()
            return pred_charges, coordinates.new_zeros(species.shape)

        # Bond vectors: coordinates is (N, 3) → (3, N) @ (N, M) = (3, M)
        bond_vecs = coordinates.T @ A
        bond_lengths = torch.linalg.norm(bond_vecs, dim=0)       # (M,)

        # Smooth, differentiable cutoff
        s = self.incline * (bond_lengths - self.displacement)    # (M,)
        cutoff_weight = torch.sigmoid(s) * self.height           # (M,)
        A_smooth = A * cutoff_weight

        # Build matrices for QEq BCI
        J = torch.diag(jii)                    # (N, N)
        K = A.T @ J @ A                        # (M, M)
        rhs = - A.T @ (chi + J @ F)            # (M,)

        # Solve 
        b = torch.linalg.pinv(K) @ rhs         # (M,)
        pred_charges = F + A @ b               # (N,)

        return pred_charges, coordinates.new_zeros(species.shape)