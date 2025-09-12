import uuid
from typing import Optional


class Neuron:
    def __init__(self, bias: Optional[float] = None):
        self.id = str(uuid.uuid4())
        self.bias = 0.0 if bias is None else bias
        self.input_sum = 0.0
        self.output = 0.0
        self.delta = 0.0

    def __repr__(self) -> str:
        return f"Neuron({self.id[:4]}, bias={self.bias:.3f})"