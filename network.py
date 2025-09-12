from typing import List
from neuron import Neuron
from layer import Layer
from connection import Connection


class Network:
    def __init__(self, layer_sizes: List[int]):
        if len(layer_sizes) < 2:
            raise ValueError("at least input and output")
        self.layers: List[Layer] = [Layer(s) for s in layer_sizes]
        self.connections: List[Connection] = []
        self._wire()

    def _wire(self):
        self.connections.clear()
        for i in range(len(self.layers) - 1):
            a = self.layers[i]
            b = self.layers[i + 1]
            for n1 in a:
                for n2 in b:
                    self.connections.append(Connection(n1, n2))

    def add_neuron(self, layer_index: int) -> Neuron:
        if not 0 <= layer_index < len(self.layers):
            raise IndexError("layer out of range")
        if layer_index == 0:
            raise ValueError("cannot add to input layer in this version")
        n = self.layers[layer_index].add_neuron()
        prev_layer = self.layers[layer_index - 1]
        for p in prev_layer:
            self.connections.append(Connection(p, n))
        if layer_index < len(self.layers) - 1:
            next_layer = self.layers[layer_index + 1]
            for q in next_layer:
                self.connections.append(Connection(n, q))
        return n

    def remove_neuron(self, layer_index: int, neuron_index: int):
        if not 0 <= layer_index < len(self.layers):
            raise IndexError("layer out of range")
        if layer_index == 0:
            raise ValueError("cannot remove from input layer")
        layer = self.layers[layer_index]
        if not 0 <= neuron_index < len(layer):
            raise IndexError("neuron out of range")
        target = layer.neurons[neuron_index]
        self.connections = [c for c in self.connections if c.source is not target and c.target is not target]
        layer.remove_neuron(neuron_index)

    def summary(self) -> str:
        parts = []
        parts.append("layers:" + ", ".join(str(len(l)) for l in self.layers))
        parts.append(f"connections:{len(self.connections)}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        return f"Network({self.summary()})"