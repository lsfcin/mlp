from typing import List, Callable, Dict
from neuron import Neuron
from layer import Layer
from connection import Connection


class Network:
    def __init__(self, layer_sizes: List[int]):
        if len(layer_sizes) < 2:
            raise ValueError("at least input and output")
        self.layers: List[Layer] = [Layer(s) for s in layer_sizes]
        self.connections: List[Connection] = []
        # ativação e índice de conexões precisam existir antes do _wire()
        self.activation_name = 'identity'
        self._activation_fn: Callable[[float], float] = lambda x: x
        self._by_target: Dict[str, List[Connection]] = {}
        self._wire()
        # estado para execução passo-a-passo
        self._fp_layer_index = 0
        self._fp_neuron_index = 0
        self._fp_inputs: List[float] = []
        # dataset (opcional) para treino/iteração
        self.dataset_inputs: List[List[float]] = []
        self.dataset_targets: List[float] = []
        self._dataset_index = 0

    def _wire(self):
        self.connections.clear()
        for i in range(len(self.layers) - 1):
            a = self.layers[i]
            b = self.layers[i + 1]
            for n1 in a:
                for n2 in b:
                    self.connections.append(Connection(n1, n2))
        self._rebuild_connection_index()

    def _rebuild_connection_index(self):
        self._by_target.clear()
        for c in self.connections:
            self._by_target.setdefault(c.target.id, []).append(c)

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

    # -------- forward propagation --------
    def reset_state(self):
        for layer in self.layers:
            for n in layer:
                n.input_sum = 0.0
                n.output = 0.0
                n.delta = 0.0
        self._fp_layer_index = 0
        self._fp_neuron_index = 0
        self._fp_inputs = []

    def start_forward(self, inputs: List[float]):
        if len(inputs) != len(self.layers[0]):
            raise ValueError("input size mismatch")
        self.reset_state()
        self._fp_inputs = list(inputs)
        # define outputs diretamente na camada de entrada
        for val, neuron in zip(inputs, self.layers[0]):
            neuron.output = val
        self._fp_layer_index = 1  # próxima camada a calcular
        self._fp_neuron_index = 0

    def forward_step(self):
        """Executa um pequeno passo do forward.

        Ordem dos passos:
        1) Se já concluído, retorna ('done', outputs)
        2) Processa um único neurônio da camada atual: acumula soma ponderada e ativa
        3) Avança para próximo neurônio ou próxima camada

        Retornos possíveis:
            ('neuron', layer_index, neuron_index, output_value)
            ('layer_done', layer_index)
            ('done', outputs_list)
        """
        # se nada iniciado ainda
        if self._fp_layer_index == 0:
            raise RuntimeError("call start_forward first")
        if self._fp_layer_index >= len(self.layers):
            # concluído
            return ('done', [n.output for n in self.layers[-1]])

        layer = self.layers[self._fp_layer_index]
        # processar neurônio atual
        if self._fp_neuron_index < len(layer):
            neuron = layer.neurons[self._fp_neuron_index]
            # somar entradas vindas da camada anterior
            prev = self.layers[self._fp_layer_index - 1]
            total = 0.0
            for c in self._by_target.get(neuron.id, []):
                # como a rede é feed-forward estrito, basta acumular
                total += c.source.output * c.weight
            total += neuron.bias
            neuron.input_sum = total
            neuron.output = self._activation_fn(total)
            result = ('neuron', self._fp_layer_index, self._fp_neuron_index, neuron.output)
            self._fp_neuron_index += 1
            # se terminou a camada, prepara próximo retorno layer_done
            if self._fp_neuron_index >= len(layer):
                # vai sinalizar layer_done no próximo passo
                return result
            return result

        # chegamos aqui quando todos neurônios da camada foram processados
        done_layer = self._fp_layer_index
        self._fp_layer_index += 1
        self._fp_neuron_index = 0
        if self._fp_layer_index >= len(self.layers):
            # finalização
            return ('done', [n.output for n in self.layers[-1]])
        return ('layer_done', done_layer)

    def forward_propagation(self, inputs: List[float]):
        """Executa forward completo (conveniência) retornando lista de outputs."""
        self.start_forward(inputs)
        while True:
            status = self.forward_step()
            if status[0] == 'done':
                return status[1]

    # ---- dataset helpers ----
    def load_dataset_rows(self, rows: List[List[str]]):
        self.dataset_inputs.clear()
        self.dataset_targets.clear()
        for r in rows:
            if len(r) < len(self.layers[0]) + 1:
                continue
            *features, target = r
            try:
                x = [float(v) for v in features[:len(self.layers[0])]]
                y = float(target)
            except ValueError:
                continue
            self.dataset_inputs.append(x)
            self.dataset_targets.append(y)
        self._dataset_index = 0

    def next_dataset_sample(self):
        if not self.dataset_inputs:
            raise RuntimeError('dataset vazio')
        x = self.dataset_inputs[self._dataset_index]
        y = self.dataset_targets[self._dataset_index]
        self._dataset_index = (self._dataset_index + 1) % len(self.dataset_inputs)
        return x, y

    # ---- backpropagation ----
    def backward(self, target: float):
        """Calcula deltas e ajusta os pesos/bias (in-place) assumindo forward já executado.

        Usa MSE: loss = 0.5 * (o - t)^2 ; d_loss/do = (o - t)
        """
        if len(self.layers[-1]) != 1:
            raise NotImplementedError('backward atual suporta apenas 1 neurônio de saída')
        output_neuron = self.layers[-1].neurons[0]
        o = output_neuron.output
        # derivada da ativação
        act = self.activation_name
        if act == 'sigmoid':
            deriv_out = o * (1 - o)
        elif act == 'relu':
            deriv_out = 1.0 if output_neuron.input_sum > 0 else 0.0
        else:  # identity
            deriv_out = 1.0
        error = o - target  # d_loss/do
        output_neuron.delta = error * deriv_out

        # camadas ocultas (reversa)
        for li in range(len(self.layers) - 2, 0, -1):  # da penúltima até a primeira oculta
            layer = self.layers[li]
            next_layer = self.layers[li + 1]
            for neuron in layer.neurons:
                # soma das contribuições das conexões de saída
                s = 0.0
                for c in self.connections:
                    if c.source is neuron and c.target in next_layer.neurons:
                        s += c.weight * c.target.delta
                # derivada ativação
                if self.activation_name == 'sigmoid':
                    deriv = neuron.output * (1 - neuron.output)
                elif self.activation_name == 'relu':
                    deriv = 1.0 if neuron.input_sum > 0 else 0.0
                else:
                    deriv = 1.0
                neuron.delta = s * deriv

        # atualizar pesos e biases
        # (usar uma taxa default, ou exigir passada externa?) => taxa default: self.learning_rate se existir, senão 0.01
        lr = getattr(self, 'learning_rate', 0.01)
        for c in self.connections:
            c.weight -= lr * c.source.output * c.target.delta
        for layer in self.layers[1:]:  # ignorar camada de entrada para bias
            for neuron in layer.neurons:
                neuron.bias -= lr * neuron.delta
        return error  # retornar erro bruto

    def train_step(self, inputs: List[float], target: float, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        outputs = self.forward_propagation(inputs)
        error = self.backward(target)
        loss = 0.5 * (error ** 2)
        return outputs, error, loss

    # ---- ativação ----
    def set_activation(self, name: str):
        name = name.lower()
        if name == 'identity':
            self._activation_fn = lambda x: x
        elif name == 'sigmoid':
            import math
            self._activation_fn = lambda x: 1.0 / (1.0 + math.exp(-x))
        elif name == 'relu':
            self._activation_fn = lambda x: x if x > 0 else 0.0
        else:
            raise ValueError('unknown activation')
        self.activation_name = name