import torch

def construct_A_matrix(D, cutoff):
    """
    Construct the A matrix using a distance cutoff on the distance matrix D.

    Parameters
    ----------
    D (N, N): Pairwise distance matrix (D[i, j] = distance between atoms i and j).
    cutoff  : Distance cutoff; any pair (i, j) with D[i, j] < cutoff is treated as a bond.

    Returns
    -------
    S (N, M).     : A matrix.
    bonds_indices : List of bonded atom index pairs (i, j) with i < j.
    """
    N = D.shape[0]
    device = D.device
    dtype = D.dtype

    # Mask of bonds based on cutoff (We only consider the upper triangle (i < j) to avoid duplicates).
    with torch.no_grad():
        cutoff_mask = (D < cutoff)
        upper_mask = torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=1)
        bond_mask = cutoff_mask & upper_mask

        # Get indices of bonds: each row is [i, j] with i < j
        bond_indices = bond_mask.nonzero(as_tuple=False)  # (M, 2) if there are M bonds

    M = bond_indices.shape[0]

    if M == 0:
        # No bonds found
        S = torch.zeros((N, 0), dtype=dtype, device=device)
        return S, []

    # bond_indices: shape [M, 2] tensor of ints
    i = bond_indices[:, 0]
    j = bond_indices[:, 1]
    e = torch.arange(len(bond_indices), device=device)
    
    A = torch.zeros((N, len(bond_indices)), dtype=dtype, device=device)
    A[i, e] = 1.0
    A[j, e] = -1.0
    
    return A, bond_indices
    
class QEqBCI(QEq):
    """
    QEq implementation with bond charge increments

    k   = A^T J A + 2 * P_diag
    bci = - k^-1 A^T (chi + J F)
    q   = F + bci
    """

    def __init__(self):
        """
        Initializes the QEqBCI class.
        """
        super().__init__()
        self._number_extra_features = 4 # Formal charges, penalty width, penalty height, hard cutoff?

    def forward(
        self,
        net_charges: torch.Tensor, #Not necessary
        features: torch.Tensor,    #Not necessary
        species: torch.Tensor,     #Not necessary
        chi: torch.Tensor,
        jii: torch.Tensor,         
        coordinates: torch.Tensor,
        F: torch.Tensor,
        sigma: torch.double,       #torch.double?
        rho: torch.double,         #torch.double?
        h_cutoff: torch.double,    #torch.double?
        interaction: int,          #Not necessary?
        num_species: int,          #Not necessary?
        rnn_count: int,            #Not necessary?
        net_spins: Optional[torch.Tensor] = None,
        ext_field: Optional[torch.Tensor] = None,
    ):
        """
        Performs QEq with BCI calculation
        """

        # Construct A
        norms = (coordinates ** 2).sum(dim=1, keepdim=True) 
        D_squared = norms + norms.T - 2 * coordinates @ coordinates.T 
        D = torch.sqrt(torch.clamp(D_squared, min=0.0)) 
        A, bonds = construct_A_matrix(D, h_cutoff)

        # Construct P
        bond_vecs = coordinates.T @ A
        bond_lengths = torch.linalg.norm(bond_vecs, dim=0)
        P = (1.0 - torch.exp(-bond_lengths / (2 * sigma**2))) * rho 

        # BCI equations
        J = torch.diag(jii) # Assuming Jii comes as a vector
        P_diag = torch.diag(P)
        K = A.T @ J @ A + 2 * P_diag
        rhs = - A.T @ (chi + J @ F)
        
        b = torch.linalg.pinv(K) @ (rhs)
        pred_charges = F + A @ b

        return pred_charges, coordinates.new_zeros(species.shape)