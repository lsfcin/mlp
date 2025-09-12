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
        # estado para execução passo-a-passo
        self._fp_layer_index = 0
        self._fp_neuron_index = 0
        self._fp_inputs: List[float] = []

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
            for c in self.connections:
                if c.target is neuron and c.source in prev.neurons:
                    total += c.source.output * c.weight
            total += neuron.bias
            neuron.input_sum = total
            # ativação simples (identidade por enquanto, pode trocar depois)
            neuron.output = total
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