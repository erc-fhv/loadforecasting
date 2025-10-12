from typing import Protocol


class ModelAdapterProtocol(Protocol):
    """Defines the minimal interface for model adapters."""
    
    def normalizeX(self, X, training=False):     
        """Normalizes X."""
        ...
        
    def normalizeY(self, Y, training=False):
        """Normalizes Y."""
        ...
        
    def deNormalizeX(self, X):
        """De-Normalizes X."""
        ...
        
    def deNormalizeY(self, Y):
        """De-Normalizes X."""
        ...
