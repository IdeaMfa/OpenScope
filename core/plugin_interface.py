from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class AnalysisPlugin(ABC):
    """
    Abstract Base Class that all analysis plugins must inherit from.
    This ensures that the GUI can interact with any algorithm uniformly.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The display name of the algorithm in the dropdown menu."""
        pass

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Default parameters for the algorithm.
        Format: {'param_name': default_value}
        Example: {'threshold': 50, 'min_size': 10}
        """
        return {}

    @abstractmethod
    def run(self, image: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Executes the analysis logic.
        
        Args:
            image: The input image (numpy array).
            params: A dictionary of parameters (user-modified).
            
        Returns:
            A mask or result image (numpy array), or None if failed.
        """
        pass