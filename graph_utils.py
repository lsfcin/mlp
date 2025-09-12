from typing import Dict, Any, Optional
from network import Network
from layer import Layer


def add_layer(network: Network, size: int, position: Optional[int] = None) -> None:
    if position is None:
        position = len(network.layers) - 1
    if position < 1 or position > len(network.layers) - 1:
        raise ValueError("posição inválida")
    from layer import Layer as _Layer
    network.layers.insert(position, _Layer(size))
    network._wire()  # type: ignore


def remove_layer(network: Network, index: int) -> None:
    if index <= 0 or index >= len(network.layers) - 1:
        raise ValueError("não é permitido remover camada de entrada ou saída")
    network.layers.pop(index)
    network._wire()  # type: ignore


def graph_data(network: Network) -> Dict[str, Any]:
    xs: Dict[int, float] = {}
    layer_spacing = 220.0
    y_spacing = 90.0
    for i, layer in enumerate(network.layers):
        xs[i] = i * layer_spacing
    nodes = []
    edges = []
    biases = [n.bias for layer in network.layers for n in layer]
    weights = [c.weight for c in network.connections]
    max_bias = max([abs(b) for b in biases] or [1.0])
    max_weight = max([abs(w) for w in weights] or [1.0])

    def color(val: float) -> str:
        if abs(val) < 1e-9:
            return "#cccccc"
        ratio = min(1.0, abs(val))
        if val >= 0:  # azul positivo
            # interpolar entre azul claro e azul escuro
            r = int(60 - 40 * ratio)
            g = int(140 - 60 * ratio)
            b = int(210 - 80 * ratio)
        else:  # vermelho negativo
            r = int(200 - 40 * (1 - ratio))
            g = int(60 - 40 * (1 - ratio))
            b = int(60 - 40 * (1 - ratio))
        return f"#{r:02x}{g:02x}{b:02x}"

    for li, layer in enumerate(network.layers):
        h = len(layer)
        for ni, neuron in enumerate(layer):
            x = xs[li]
            offset = -(h - 1) / 2.0
            y = (offset + ni) * y_spacing
            norm = abs(neuron.bias) / max_bias if max_bias else 0.0
            width = 2.0 + 8.0 * norm
            nodes.append({
                "id": neuron.id,
                "bias": neuron.bias,
                "layer": li,
                "x": x,
                "y": y,
                "color": color(neuron.bias),
                "width": width,
                "label": f"b={neuron.bias:.2f}",
            })
    for c in network.connections:
        norm = abs(c.weight) / max_weight if max_weight else 0.0
        width = 1.0 + 6.0 * norm
        edges.append({
            "id": c.id,
            "source": c.source.id,
            "target": c.target.id,
            "weight": c.weight,
            "color": color(c.weight),
            "width": width,
            "label": f"w={c.weight:.2f}",
        })
    return {"nodes": nodes, "edges": edges}
