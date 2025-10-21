"""
This module specifies all parameters of the prediction models.
"""
from typing import Protocol
import torch

class ModelAdapterProtocol(Protocol):
    """Defines the minimal interface for model adapters."""

    def normalize_x(
        self,
        x: torch.Tensor,
        training: bool = False,
        ) -> torch.Tensor:
        """Normalizes x."""
        ...

    def normalize_y(
        self,
        y: torch.Tensor,
        training: bool = False,
        ) -> torch.Tensor:
        """Normalizes y."""
        ...

    def de_normalize_x(
        self,
        x: torch.Tensor,
        ) -> torch.Tensor:
        """De-Normalizes X."""
        ...

    def de_normalize_y(
        self,
        y: torch.Tensor,
        ) -> torch.Tensor:
        """De-Normalizes X."""
        ...

