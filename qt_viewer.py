from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import (
    QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem,
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSpinBox, QLabel, QFrame, QMessageBox
)
from PyQt6.QtGui import QBrush, QPen, QColor, QTransform, QPainter
from PyQt6.QtCore import Qt, QPointF, QTimer

from network import Network
from graph_utils import graph_data, add_layer, remove_layer


class GraphNodeItem(QGraphicsEllipseItem):
    def __init__(self, node: Dict[str, Any]):
        size = node['width'] * 16.0  # 4x maior que antes
        super().__init__(-size/2, -size/2, size, size)
        self.node_id = node['id']
        self.bias = node.get('bias', 0.0)
        self.setBrush(QBrush(QColor(node['color'])))
        pen = QPen(QColor(node['color']))
        pen.setWidthF(max(1.0, node['width']))
        self.setPen(pen)
        self.setToolTip(f"bias={self.bias:.4f}")
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setPos(QPointF(node['x'], node['y']))
        # label de bias abaixo
        self.label_item = QGraphicsTextItem(f"{self.bias:.2f}")
        self.label_item.setDefaultTextColor(QColor('#222222'))
        self.label_item.setPos(-self.label_item.boundingRect().width()/2, size/2 + 4)
        self.label_item.setParentItem(self)
        self._base_brush = self.brush()
        self._highlight = False

    def highlight(self, on: bool):
        if on == self._highlight:
            return
        self._highlight = on
        if on:
            b = QBrush(QColor('#ffff99'))
            self.setBrush(b)
        else:
            self.setBrush(self._base_brush)


class GraphEdgeItem(QGraphicsLineItem):
    def __init__(self, src: Dict[str, Any], dst: Dict[str, Any], edge: Dict[str, Any]):
        super().__init__(src['x'], src['y'], dst['x'], dst['y'])
        pen = QPen(QColor(edge['color']))
        pen.setWidthF(max(0.5, edge['width']))
        self.setPen(pen)
        self.setToolTip(edge['label'])
        self.setZValue(-1)
        # label de peso no meio
        mx = (src['x'] + dst['x']) / 2.0
        my = (src['y'] + dst['y']) / 2.0
        self.weight_label = QGraphicsTextItem(f"{edge['weight']:.2f}")
        self.weight_label.setDefaultTextColor(QColor('#111111'))
        self.weight_label.setPos(mx, my)
        self.weight_label.setZValue(10)


class GraphScene(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.setBackgroundBrush(QBrush(QColor('#ffffff')))
        self.node_items: List[GraphNodeItem] = []
        self._input_value_labels: List[QGraphicsTextItem] = []
        self._output_value_label: Optional[QGraphicsTextItem] = None

    def build(self, network: Network):
        self.clear()
        self.node_items.clear()
        self._input_value_labels.clear()
        self._output_value_label = None
        data = graph_data(network)
        nodes_index = {n['id']: n for n in data['nodes']}
        # edges
        for e in data['edges']:
            src = nodes_index[e['source']]
            dst = nodes_index[e['target']]
            edge_item = GraphEdgeItem(src, dst, e)
            self.addItem(edge_item)
            self.addItem(edge_item.weight_label)
        # nodes
        node_items_raw = []
        for n in data['nodes']:
            item = GraphNodeItem(n)
            node_items_raw.append((n, item))
            self.node_items.append(item)
            self.addItem(item)
        # labels de entrada e nomes de features
        if node_items_raw:
            input_layer_nodes = [pair for pair in node_items_raw if pair[0]['layer'] == 0]
            feature_names = self._default_feature_names(len(input_layer_nodes))
            for idx, (raw, item) in enumerate(input_layer_nodes):
                left_label = QGraphicsTextItem("0")
                left_label.setDefaultTextColor(QColor('#444444'))
                br = left_label.boundingRect()
                left_label.setPos(item.x() - item.rect().width()/2 - br.width() - 12, item.y() - br.height()/2)
                self.addItem(left_label)
                self._input_value_labels.append(left_label)
                feat_label = QGraphicsTextItem(feature_names[idx])
                feat_label.setDefaultTextColor(QColor('#000000'))
                fr = feat_label.boundingRect()
                feat_label.setPos(item.x() - fr.width()/2, item.y() - item.rect().height()/2 - fr.height() - 6)
                self.addItem(feat_label)
        # label de saída
        output_layer_index = max(n['layer'] for n, _ in node_items_raw) if node_items_raw else 0
        output_nodes = [pair for pair in node_items_raw if pair[0]['layer'] == output_layer_index]
        if output_nodes:
            raw, item = output_nodes[-1]
            out_label = QGraphicsTextItem("0")
            out_label.setDefaultTextColor(QColor('#222222'))
            orc = out_label.boundingRect()
            out_label.setPos(item.x() + item.rect().width()/2 + 12, item.y() - orc.height()/2)
            self.addItem(out_label)
            self._output_value_label = out_label
        self.setSceneRect(self.itemsBoundingRect().adjusted(-120, -120, 120, 120))

    def _default_feature_names(self, count: int) -> List[str]:
        base = [
            'idade','sexo','cp','pressao','colesterol','acucar','ecg','freq','angina','oldpeak','slope','vessels','thal'
        ]
        if count <= len(base):
            return base[:count]
        extra = [f'f{i}' for i in range(len(base), count)]
        return base + extra

    def highlight_neuron(self, layer_index: int, neuron_index: int, network: Network):
        # remove highlight anterior
        for item in self.node_items:
            item.highlight(False)
        # localizar neurônio alvo pelos índices
        if not (0 <= layer_index < len(network.layers)):
            return
        layer = network.layers[layer_index]
        if not (0 <= neuron_index < len(layer)):
            return
        target_id = layer.neurons[neuron_index].id
        for item in self.node_items:
            if getattr(item, 'node_id', None) == target_id:
                item.highlight(True)
                break

    def update_input_values(self, values: List[float]):
        for val, label in zip(values, self._input_value_labels):
            label.setPlainText(f"{val:.2f}")

    def update_output_value(self, value: float):
        if self._output_value_label:
            self._output_value_label.setPlainText(f"{value:.2f}")


class GraphView(QGraphicsView):
    def __init__(self, scene: GraphScene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.scale_factor = 1.15

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scale(self.scale_factor, self.scale_factor)
        else:
            self.scale(1 / self.scale_factor, 1 / self.scale_factor)


class NetworkViewerWindow(QMainWindow):
    def __init__(self, network: Network):
        super().__init__()
        self.network = network
        self.setWindowTitle('Visualizador de Rede Neural (PyQt)')
        self.scene = GraphScene()
        self.view = GraphView(self.scene)
        self._forward_started = False
        self._playing = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.view, stretch=4)
        side = QVBoxLayout()
        layout.addLayout(side, stretch=1)

        # Layer controls
        side.addWidget(self._section_label('Camadas'))
        h_add_layer = QHBoxLayout()
        self.spin_add_layer_size = QSpinBox(); self.spin_add_layer_size.setMinimum(1); self.spin_add_layer_size.setValue(2)
        self.spin_add_layer_pos = QSpinBox(); self.spin_add_layer_pos.setMinimum(1)
        btn_add_layer = QPushButton('Adicionar Camada')
        btn_add_layer.clicked.connect(self._on_add_layer)
        h_add_layer.addWidget(QLabel('Tamanho'))
        h_add_layer.addWidget(self.spin_add_layer_size)
        h_add_layer.addWidget(QLabel('Posição'))
        h_add_layer.addWidget(self.spin_add_layer_pos)
        side.addLayout(h_add_layer)
        side.addWidget(btn_add_layer)

        h_remove_layer = QHBoxLayout()
        self.spin_remove_layer = QSpinBox(); self.spin_remove_layer.setMinimum(1)
        btn_remove_layer = QPushButton('Remover Camada')
        btn_remove_layer.clicked.connect(self._on_remove_layer)
        h_remove_layer.addWidget(QLabel('Índice'))
        h_remove_layer.addWidget(self.spin_remove_layer)
        side.addLayout(h_remove_layer)
        side.addWidget(btn_remove_layer)

        side.addWidget(self._separator())

        # Neuron controls
        side.addWidget(self._section_label('Neurônios'))
        h_add_neuron = QHBoxLayout()
        self.spin_add_neuron_layer = QSpinBox(); self.spin_add_neuron_layer.setMinimum(1)
        btn_add_neuron = QPushButton('Adicionar Neurônio')
        btn_add_neuron.clicked.connect(self._on_add_neuron)
        h_add_neuron.addWidget(QLabel('Camada'))
        h_add_neuron.addWidget(self.spin_add_neuron_layer)
        side.addLayout(h_add_neuron)
        side.addWidget(btn_add_neuron)

        h_remove_neuron = QHBoxLayout()
        self.spin_remove_neuron_layer = QSpinBox(); self.spin_remove_neuron_layer.setMinimum(1)
        self.spin_remove_neuron_index = QSpinBox(); self.spin_remove_neuron_index.setMinimum(0)
        btn_remove_neuron = QPushButton('Remover Neurônio')
        btn_remove_neuron.clicked.connect(self._on_remove_neuron)
        h_remove_neuron.addWidget(QLabel('Camada'))
        h_remove_neuron.addWidget(self.spin_remove_neuron_layer)
        h_remove_neuron.addWidget(QLabel('Índice'))
        h_remove_neuron.addWidget(self.spin_remove_neuron_index)
        side.addLayout(h_remove_neuron)
        side.addWidget(btn_remove_neuron)

        side.addWidget(self._separator())

        # Playback controls (stubs)
        side.addWidget(self._section_label('Execução'))
        h_play = QHBoxLayout()
        self.btn_play = QPushButton('▶')
        self.btn_step = QPushButton('Passo')
        self.btn_pause = QPushButton('⏸')
        self.btn_reset = QPushButton('Reset')
        for b in (self.btn_play, self.btn_step, self.btn_pause, self.btn_reset):
            h_play.addWidget(b)
        self.btn_play.clicked.connect(self._on_play)
        self.btn_step.clicked.connect(self._on_step)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_reset.clicked.connect(self._on_reset)
        side.addLayout(h_play)

        self.label_status = QLabel('Pronto')
        side.addWidget(self.label_status)
        side.addStretch(1)

        self.setCentralWidget(root)

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet('font-weight: bold; margin-top:8px;')
        return lbl

    def _separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet('color:#999;')
        return line

    def _refresh(self):
        self.scene.build(self.network)
        self.label_status.setText(f"Camadas: {[len(l) for l in self.network.layers]}")
        # se já havia forward iniciado, re-sincronizar inputs
        if self._forward_started:
            self.scene.update_input_values(self.network._fp_inputs)

    # Event handlers
    def _on_add_layer(self):
        try:
            size = self.spin_add_layer_size.value()
            pos = self.spin_add_layer_pos.value()
            add_layer(self.network, size, pos)
            self._refresh()
        except Exception as e:
            QMessageBox.warning(self, 'Erro', str(e))

    def _on_remove_layer(self):
        try:
            idx = self.spin_remove_layer.value()
            remove_layer(self.network, idx)
            self._refresh()
        except Exception as e:
            QMessageBox.warning(self, 'Erro', str(e))

    def _on_add_neuron(self):
        try:
            layer_index = self.spin_add_neuron_layer.value()
            self.network.add_neuron(layer_index)
            self._refresh()
        except Exception as e:
            QMessageBox.warning(self, 'Erro', str(e))

    def _on_remove_neuron(self):
        try:
            layer_index = self.spin_remove_neuron_layer.value()
            neuron_index = self.spin_remove_neuron_index.value()
            self.network.remove_neuron(layer_index, neuron_index)
            self._refresh()
        except Exception as e:
            QMessageBox.warning(self, 'Erro', str(e))

    # Forward control handlers
    def _ensure_forward_started(self):
        if not self._forward_started:
            # gerar inputs simples (ex: zeros) tamanho camada entrada
            inputs = [0.0 for _ in self.network.layers[0]]
            self.network.start_forward(inputs)
            self.scene.update_input_values(inputs)
            self._forward_started = True
            self.label_status.setText('Forward iniciado')

    def _on_step(self):
        self._ensure_forward_started()
        status = self.network.forward_step()
        kind = status[0]
        if kind == 'neuron':
            _, layer_i, neuron_i, value = status
            self.scene.highlight_neuron(layer_i, neuron_i, self.network)
            self.label_status.setText(f"Neuron ({layer_i},{neuron_i}) = {value:.4f}")
            if layer_i == len(self.network.layers) - 1:
                # atualizar saída parcial
                self.scene.update_output_value(value)
        elif kind == 'layer_done':
            _, layer_i = status
            self.label_status.setText(f"Camada {layer_i} concluída")
        elif kind == 'done':
            _, outputs = status
            out_val = outputs[0] if outputs else 0.0
            self.scene.update_output_value(out_val)
            self.label_status.setText(f"Concluído. Output={out_val:.4f}")
            self._playing = False
            self._play_timer.stop()

    def _on_play(self):
        if self._playing:
            return
        self._ensure_forward_started()
        self._playing = True
        self._play_timer.start(300)  # ms
        self.label_status.setText('Reproduzindo...')

    def _on_play_tick(self):
        if not self._playing:
            return
        self._on_step()

    def _on_pause(self):
        if self._playing:
            self._playing = False
            self._play_timer.stop()
            self.label_status.setText('Pausado')

    def _on_reset(self):
        self._playing = False
        self._play_timer.stop()
        self.network.reset_state()
        self._forward_started = False
        self.scene.build(self.network)
        self.label_status.setText('Reset')


def launch_qt_viewer():
    import sys
    app = QApplication(sys.argv)
    net = Network([13, 1])
    win = NetworkViewerWindow(net)
    win.resize(1200, 700)
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    launch_qt_viewer()
