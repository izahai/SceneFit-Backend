import torch
from PIL import Image
from typing import List, Optional

class ClipSubspace:
    def __init__(self, basis: torch.Tensor):
        """
        basis: [d, k] orthonormal columns
        """
        self.basis = basis

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project x into the subspace
        x: [n, d] or [d]
        """
        basis = self.basis.to(dtype=x.dtype, device=x.device)

        if x.dim() == 1:
            return basis @ (basis.T @ x)
        return x @ basis @ basis.T

    def remove(self, x: torch.Tensor) -> torch.Tensor:
        """
        Remove subspace component
        """
        return x - self.project(x)

def orthogonalize_subspaces(
    primary: ClipSubspace,
    secondary: ClipSubspace,
) -> ClipSubspace:
    """
    Make secondary orthogonal to primary
    """
    B = primary.basis
    E = secondary.basis

    # Remove primary directions from secondary
    E_ortho = E - B @ (B.T @ E)

    # Re-orthonormalize
    Q, _ = torch.linalg.qr(E_ortho)

    return ClipSubspace(Q)
