import uuid
import random
from typing import Optional

from neuron import Neuron


class Connection:
    def __init__(self, a: Neuron, b: Neuron, weight: Optional[float] = None):
        self.id = str(uuid.uuid4())
        self.source = a
        self.target = b
        self.weight = random.uniform(-1.0, 1.0) if weight is None else weight

    def __repr__(self) -> str:
        return f"Connection({self.source.id[:4]}->{self.target.id[:4]}, w={self.weight:.3f})"
