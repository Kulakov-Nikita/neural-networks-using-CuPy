from abc import ABC, abstractmethod
import cupy as cp

class Layer(ABC):
    @abstractmethod
    def forward(self, x: cp.ndarray) -> cp.ndarray:
        pass

    @abstractmethod
    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        pass