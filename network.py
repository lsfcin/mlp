from typing import List, Callable, Dict, Optional
import random
import json
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
        # gradientes da última atualização (por conexão id)
        self.last_gradients = {}
        # preprocessamento
        self.preprocess_mode = 'none'  # 'none' | 'normalize' | 'standardize'
        self._feat_min: List[float] = []
        self._feat_max: List[float] = []
        self._feat_mean: List[float] = []
        self._feat_std: List[float] = []

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
    def reset_forward_state(self):
        """Reseta apenas o estado de execução do forward (sem tocar em pesos/biases)."""
        for layer in self.layers:
            for n in layer:
                n.input_sum = 0.0
                n.output = 0.0
                n.delta = 0.0
        self._fp_layer_index = 0
        self._fp_neuron_index = 0
        self._fp_inputs = []

    def reset_state(self):
        """Reset completo: estado de forward + re-inicialização de pesos e vieses."""
        self.reset_forward_state()
        # também resetar pesos e vieses conforme solicitado
        self._randomize_parameters()

    def _randomize_parameters(self):
        for c in self.connections:
            c.weight = random.uniform(-1.0, 1.0)
        for layer in self.layers[1:]:  # ignorar camada de entrada
            for n in layer:
                n.bias = 0.0

    def start_forward(self, inputs: List[float]):
        if len(inputs) != len(self.layers[0]):
            raise ValueError("input size mismatch")
        # não randomizar pesos ao iniciar um forward; apenas limpar estado
        self.reset_forward_state()
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
        # calcular estatísticas para preprocessamento
        d = len(self.layers[0])
        if self.dataset_inputs:
            cols = list(zip(*self.dataset_inputs))
            self._feat_min = [min(col) for col in cols]
            self._feat_max = [max(col) for col in cols]
            self._feat_mean = [sum(col) / len(col) for col in cols]
            self._feat_std = []
            for i, col in enumerate(cols):
                m = self._feat_mean[i]
                var = sum((v - m) ** 2 for v in col) / max(1, (len(col)))
                std = var ** 0.5
                if std < 1e-12:
                    std = 1.0
                self._feat_std.append(std)

    def next_dataset_sample(self):
        if not self.dataset_inputs:
            raise RuntimeError('dataset vazio')
        x = self.transform_inputs(self.dataset_inputs[self._dataset_index])
        y = self.dataset_targets[self._dataset_index]
        self._dataset_index = (self._dataset_index + 1) % len(self.dataset_inputs)
        return x, y

    # ---- preprocessamento ----
    def set_preprocess_mode(self, mode: str):
        mode = mode.lower()
        if mode not in ('none', 'normalize', 'standardize'):
            raise ValueError('modo inválido de preprocessamento')
        self.preprocess_mode = mode

    def transform_inputs(self, inputs: List[float]) -> List[float]:
        if self.preprocess_mode == 'none':
            return list(inputs)
        if not self._feat_min or len(self._feat_min) != len(inputs):
            return list(inputs)
        out = []
        if self.preprocess_mode == 'normalize':
            for v, mn, mx in zip(inputs, self._feat_min, self._feat_max):
                rng = mx - mn
                if abs(rng) < 1e-12:
                    out.append(0.0)
                else:
                    out.append((v - mn) / rng)
        elif self.preprocess_mode == 'standardize':
            for v, m, s in zip(inputs, self._feat_mean, self._feat_std):
                out.append((v - m) / (s if s else 1.0))
        else:
            out = list(inputs)
        return out

    # ---- backpropagation ----
    def _compute_deltas(self, target: float) -> float:
        """Calcula deltas (gradientes de erro) sem atualizar pesos. Retorna erro bruto (o - target)."""
        if len(self.layers[-1]) != 1:
            raise NotImplementedError('backward atual suporta apenas 1 neurônio de saída')
        output_neuron = self.layers[-1].neurons[0]
        o = output_neuron.output
        if self.activation_name == 'sigmoid':
            deriv_out = o * (1 - o)
        elif self.activation_name == 'relu':
            deriv_out = 1.0 if output_neuron.input_sum > 0 else 0.0
        else:
            deriv_out = 1.0
        error = o - target
        output_neuron.delta = error * deriv_out
        # ocultas
        for li in range(len(self.layers) - 2, 0, -1):
            layer = self.layers[li]
            next_layer = self.layers[li + 1]
            for neuron in layer.neurons:
                s = 0.0
                for c in self.connections:
                    if c.source is neuron and c.target in next_layer.neurons:
                        s += c.weight * c.target.delta
                if self.activation_name == 'sigmoid':
                    deriv = neuron.output * (1 - neuron.output)
                elif self.activation_name == 'relu':
                    deriv = 1.0 if neuron.input_sum > 0 else 0.0
                else:
                    deriv = 1.0
                neuron.delta = s * deriv
        return error

    def backward(self, target: float):
        """Executa passo de backprop completo (deltas + atualização de pesos). Retorna erro bruto."""
        error = self._compute_deltas(target)
        lr = getattr(self, 'learning_rate', 0.01)
        for c in self.connections:
            c.weight -= lr * c.source.output * c.target.delta
        for layer in self.layers[1:]:
            for neuron in layer.neurons:
                neuron.bias -= lr * neuron.delta
        return error

    def train_step(self, inputs: List[float], target: float, learning_rate: float = 0.01):
        """Executa um passo de treino (forward + backprop) armazenando gradientes.

        Retorna (outputs, erro_bruto, loss).
        last_gradients: dict connection_id -> gradiente (dL/dw) ANTES da atualização.
        """
        self.learning_rate = learning_rate
        outputs = self.forward_propagation(self.transform_inputs(inputs))
        error = self._compute_deltas(target)
        # coletar gradientes antes de aplicar atualização
        grads: Dict[str, float] = {}
        for c in self.connections:
            grads[c.id] = c.source.output * c.target.delta  # dL/dw
        self.last_gradients = grads
        # aplicar update
        lr = self.learning_rate
        for c in self.connections:
            c.weight -= lr * grads[c.id]
        for layer in self.layers[1:]:
            for neuron in layer.neurons:
                neuron.bias -= lr * neuron.delta  # grad bias = delta
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

    # ---- persistência ----
    def to_dict(self) -> Dict:
        sizes = [len(l) for l in self.layers]
        data: Dict[str, Optional[Dict]] = {
            'layers': sizes,
            'activation': self.activation_name,
            'biases': {},
            'weights': {},
        }
        for li in range(1, len(self.layers)):
            # biases da camada li
            data['biases'][str(li)] = [n.bias for n in self.layers[li].neurons]
            # pesos da conexão layer li-1 -> li
            prev = self.layers[li - 1]
            cur = self.layers[li]
            mat: List[List[float]] = []
            for pj, pn in enumerate(prev.neurons):
                row = []
                for cj, cn in enumerate(cur.neurons):
                    # achar conexão pn -> cn
                    w = 0.0
                    for c in self.connections:
                        if c.source is pn and c.target is cn:
                            w = c.weight
                            break
                    row.append(w)
                mat.append(row)
            data['weights'][str(li)] = mat
        return data

    def save_json(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def load_json(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        sizes = data.get('layers')
        if sizes and sizes != [len(l) for l in self.layers]:
            # reconstruir rede para bater os tamanhos
            self.layers = [Layer(s) for s in sizes]
            self._wire()
        # aplicar biases e pesos
        for li in range(1, len(self.layers)):
            # biases
            b_list = data.get('biases', {}).get(str(li))
            if b_list and len(b_list) == len(self.layers[li]):
                for n, b in zip(self.layers[li].neurons, b_list):
                    n.bias = float(b)
            # pesos
            mat = data.get('weights', {}).get(str(li))
            if mat:
                prev = self.layers[li - 1]
                cur = self.layers[li]
                for pj, pn in enumerate(prev.neurons):
                    for cj, cn in enumerate(cur.neurons):
                        val = float(mat[pj][cj])
                        for c in self.connections:
                            if c.source is pn and c.target is cn:
                                c.weight = val
                                break
        act = data.get('activation')
        if act:
            self.set_activation(act)