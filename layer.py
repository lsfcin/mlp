from typing import List, Optional
from neuron import Neuron


class Layer:
    def __init__(self, size: int):
        self.neurons: List[Neuron] = [Neuron() for _ in range(size)]

    def add_neuron(self, neuron: Optional[Neuron] = None) -> Neuron:
        n = neuron or Neuron()
        self.neurons.append(n)
        return n

    def remove_neuron(self, index: int) -> Neuron:
        return self.neurons.pop(index)

    def __len__(self) -> int:
        return len(self.neurons)

    def __iter__(self):
        return iter(self.neurons)

    def __repr__(self) -> str:
        return f"Layer(size={len(self.neurons)})"