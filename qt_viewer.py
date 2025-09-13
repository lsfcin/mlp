from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import (
    QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem, QGraphicsPolygonItem,
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSpinBox, QLabel, QFrame, QMessageBox, QLineEdit, QComboBox, QDoubleSpinBox, QPlainTextEdit
)
from PyQt6.QtGui import QBrush, QPen, QColor, QTransform, QPainter, QPolygonF
from PyQt6.QtCore import Qt, QPointF, QTimer

from network import Network
from graph_utils import graph_data, add_layer, remove_layer
from loader import Loader


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
        self.edge_id = edge['id']
        pen = QPen(QColor(edge['color']))
        pen.setWidthF(max(0.5, edge['width']))
        self.setPen(pen)
        self.setToolTip(edge['label'])
        self.setZValue(-2)
        # label de peso no meio
        mx = (src['x'] + dst['x']) / 2.0
        my = (src['y'] + dst['y']) / 2.0
        self.weight_label = QGraphicsTextItem(f"{edge['weight']:.2f}")
        self.weight_label.setDefaultTextColor(QColor('#111111'))
        self.weight_label.setPos(mx, my)
        self.weight_label.setZValue(10)
        self.weight_label.setVisible(False)
        # seta (triângulo) perto do destino
        dx = dst['x'] - src['x']
        dy = dst['y'] - src['y']
        length = (dx**2 + dy**2) ** 0.5 or 1.0
        ux, uy = dx / length, dy / length
        arrow_size = 10
        tip_x = dst['x'] - ux * 18
        tip_y = dst['y'] - uy * 18
        left_x = tip_x - uy * arrow_size/2 - ux * arrow_size
        left_y = tip_y + ux * arrow_size/2 - uy * arrow_size
        right_x = tip_x + uy * arrow_size/2 - ux * arrow_size
        right_y = tip_y - ux * arrow_size/2 - uy * arrow_size
        poly = QPolygonF([
            QPointF(tip_x, tip_y),
            QPointF(left_x, left_y),
            QPointF(right_x, right_y)
        ])
        self.arrow_item = QGraphicsPolygonItem(poly)
        self.arrow_item.setBrush(QBrush(QColor(edge['color'])))
        self.arrow_item.setPen(QPen(Qt.PenStyle.NoPen))
        self.arrow_item.setZValue(-1)
        # hover events
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event):
        if hasattr(self, 'weight_label'):
            self.weight_label.setVisible(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        if hasattr(self, 'weight_label'):
            self.weight_label.setVisible(False)
        super().hoverLeaveEvent(event)


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
        self.edge_items: List[GraphEdgeItem] = []
        for e in data['edges']:
            src = nodes_index[e['source']]
            dst = nodes_index[e['target']]
            edge_item = GraphEdgeItem(src, dst, e)
            self.addItem(edge_item)
            self.addItem(edge_item.weight_label)
            self.addItem(edge_item.arrow_item)
            self.edge_items.append(edge_item)
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
        # highlight conexões de entrada
        self.highlight_connections_to(neuron=layer.neurons[neuron_index], network=network)

    def highlight_connections_to(self, neuron, network: Network):
        # recolor tudo base
        for e in self.edge_items:
            pen = e.pen(); color = pen.color(); color.setAlpha(120); pen.setColor(color); e.setPen(pen)
            e.arrow_item.setOpacity(0.25)
        # obter conexões que chegam a este neurônio
        for c in network._by_target.get(neuron.id, []):
            for e in self.edge_items:
                if hasattr(e, 'weight_label') and c.source.id in e.toolTip() and c.target.id in e.toolTip():
                    pen = e.pen(); pen.setColor(QColor('#222222')); pen.setWidthF(pen.widthF()+1); e.setPen(pen)
                    e.arrow_item.setOpacity(0.9)

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


class LossChartWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.history: List[float] = []
        self.setMinimumHeight(120)

    def set_history(self, values: List[float]):
        self.history = list(values)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#fafafa'))
        if not self.history:
            painter.end()
            return
        w = self.width(); h = self.height()
        margin = 8
        chart_w = max(1, w - 2*margin)
        chart_h = max(1, h - 2*margin)
        min_v = min(self.history)
        max_v = max(self.history)
        rng = max(1e-9, (max_v - min_v))
        pen_axis = QPen(QColor('#cccccc'))
        painter.setPen(pen_axis)
        painter.drawRect(margin, margin, chart_w, chart_h)
        pen_line = QPen(QColor('#0070f3'))
        pen_line.setWidth(2)
        painter.setPen(pen_line)
        n = len(self.history)
        for i in range(1, n):
            x1 = margin + (i-1) * chart_w / max(1, n-1)
            y1 = margin + chart_h - ((self.history[i-1] - min_v) / rng) * chart_h
            x2 = margin + i * chart_w / max(1, n-1)
            y2 = margin + chart_h - ((self.history[i] - min_v) / rng) * chart_h
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        painter.end()


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
        self._dataset_loaded = False
        self._current_target = None
        self._loss_history: List[float] = []
        self._loss_window = None  # lazy

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

        # Entrada customizada
        side.addWidget(self._section_label('Entrada'))
        h_input = QHBoxLayout()
        self.edit_inputs = QLineEdit()
        self.edit_inputs.setPlaceholderText('Ex: 0.1, -0.2, 1.0, ...')
        h_input.addWidget(self.edit_inputs)
        btn_set_inputs = QPushButton('Aplicar')
        btn_set_inputs.clicked.connect(self._on_apply_inputs)
        h_input.addWidget(btn_set_inputs)
        side.addLayout(h_input)

        # Ativação
        h_act = QHBoxLayout()
        h_act.addWidget(QLabel('Ativação'))
        self.combo_activation = QComboBox()
        self.combo_activation.addItems(['identity','sigmoid','relu'])
        self.combo_activation.currentTextChanged.connect(self._on_change_activation)
        h_act.addWidget(self.combo_activation)
        side.addLayout(h_act)

        # Dataset
        side.addWidget(self._separator())
        side.addWidget(self._section_label('Dataset'))
        h_ds = QHBoxLayout()
        self.btn_load_ds = QPushButton('Carregar heart.csv')
        self.btn_load_ds.clicked.connect(self._on_load_dataset)
        h_ds.addWidget(self.btn_load_ds)
        self.btn_next_sample = QPushButton('Próximo sample')
        self.btn_next_sample.clicked.connect(self._on_next_sample)
        h_ds.addWidget(self.btn_next_sample)
        side.addLayout(h_ds)
        self.btn_fast_forward = QPushButton('Fast Forward (todas)')
        self.btn_fast_forward.clicked.connect(self._on_fast_forward)
        side.addWidget(self.btn_fast_forward)

        # Pré-processamento
        h_prep = QHBoxLayout()
        h_prep.addWidget(QLabel('Pré-processamento'))
        self.combo_prep = QComboBox()
        self.combo_prep.addItems(['none', 'normalize', 'standardize'])
        self.combo_prep.setCurrentText(getattr(self.network, 'preprocess_mode', 'none'))
        self.combo_pep_current_changed = self.combo_prep.currentTextChanged.connect(self._on_change_prep)
        h_prep.addWidget(self.combo_prep)
        side.addLayout(h_prep)

        # Persistência
        h_persist = QHBoxLayout()
        self.btn_save = QPushButton('Salvar JSON')
        self.btn_save.clicked.connect(self._on_save_weights)
        self.btn_load = QPushButton('Carregar JSON')
        self.btn_load.clicked.connect(self._on_load_weights)
        h_persist.addWidget(self.btn_save)
        h_persist.addWidget(self.btn_load)
        side.addLayout(h_persist)

        side.addWidget(self._separator())
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

        # Treino
        side.addWidget(self._separator())
        side.addWidget(self._section_label('Treino'))
        h_lr = QHBoxLayout()
        h_lr.addWidget(QLabel('LR'))
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(4)
        self.spin_lr.setSingleStep(0.001)
        self.spin_lr.setRange(0.0001, 10.0)
        self.spin_lr.setValue(0.01)
        h_lr.addWidget(self.spin_lr)
        side.addLayout(h_lr)

        h_train_btns = QHBoxLayout()
        self.btn_train_step = QPushButton('Train Step')
        self.btn_train_epoch = QPushButton('Train Época')
        h_train_btns.addWidget(self.btn_train_step)
        h_train_btns.addWidget(self.btn_train_epoch)
        side.addLayout(h_train_btns)
        self.btn_train_step.clicked.connect(self._on_train_step)
        self.btn_train_epoch.clicked.connect(self._on_train_epoch)

        self.btn_show_loss = QPushButton('Ver Loss')
        self.btn_show_loss.clicked.connect(self._on_show_loss)
        side.addWidget(self.btn_show_loss)
        # Gráfico inline de Loss
        self.loss_chart = LossChartWidget()
        side.addWidget(self.loss_chart)
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
            if self.edit_inputs.text().strip():
                inputs = self._parse_inputs_text()
            elif self._dataset_loaded:
                x, y = self.network.next_dataset_sample()
                inputs = x
                self._current_target = y
            else:
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
            # atualizar cor do nó pela saída
            self._update_node_colors()
        elif kind == 'layer_done':
            _, layer_i = status
            self.label_status.setText(f"Camada {layer_i} concluída")
        elif kind == 'done':
            _, outputs = status
            out_val = outputs[0] if outputs else 0.0
            self.scene.update_output_value(out_val)
            # erro se target presente
            if hasattr(self, '_current_target') and self._current_target is not None:
                err = out_val - self._current_target
                self.label_status.setText(f"Concluído. Output={out_val:.4f} Erro={err:.4f}")
            else:
                self.label_status.setText(f"Concluído. Output={out_val:.4f}")
            self._playing = False
            self._play_timer.stop()
            self._update_node_colors()

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
        self._current_target = None

    # Dataset handlers
    def _on_load_dataset(self):
        try:
            loader = Loader('rsc/heart.csv')
            self.network.load_dataset_rows(loader.rows)
            if self.network.dataset_inputs:
                self._dataset_loaded = True
                self.label_status.setText(f'Dataset carregado: {len(self.network.dataset_inputs)} samples')
            else:
                self.label_status.setText('Dataset vazio / inválido')
        except Exception as e:
            QMessageBox.warning(self, 'Erro', f'Falha ao carregar dataset: {e}')

    def _on_next_sample(self):
        if not self._dataset_loaded:
            QMessageBox.information(self, 'Info', 'Carregue o dataset primeiro.')
            return
        if self._forward_started:
            QMessageBox.information(self, 'Info', 'Reset antes de pegar novo sample.')
            return
        x, y = self.network.next_dataset_sample()
        self.scene.update_input_values(x)
        self.edit_inputs.setText(', '.join(f'{v:.3f}' for v in x))
        self._current_target = y
        self.label_status.setText('Sample pronto (target armazenado)')

    def _on_fast_forward(self):
        if not self._dataset_loaded:
            QMessageBox.information(self, 'Info', 'Carregue o dataset primeiro.')
            return
        if self._forward_started:
            QMessageBox.information(self, 'Info', 'Use Reset antes de fast-forward.')
            return
        total = len(self.network.dataset_inputs)
        correct = 0
        for x, target in zip(self.network.dataset_inputs, self.network.dataset_targets):
            outputs = self.network.forward_propagation(x)
            pred = outputs[0]
            # classificação binária simples threshold 0.5 (para sigmoid/identity)
            if (pred >= 0.5 and target >= 0.5) or (pred < 0.5 and target < 0.5):
                correct += 1
        acc = correct / total if total else 0.0
        self.label_status.setText(f'FastForward concluído acc={acc:.3f}')

    def _parse_inputs_text(self) -> List[float]:
        txt = self.edit_inputs.text().strip()
        if not txt:
            return [0.0 for _ in self.network.layers[0]]
        parts = [p.strip() for p in txt.replace(';', ',').split(',') if p.strip()]
        vals = []
        for p in parts:
            try:
                vals.append(float(p))
            except ValueError:
                raise ValueError(f"valor inválido: {p}")
        if len(vals) != len(self.network.layers[0]):
            raise ValueError(f"esperado {len(self.network.layers[0])} valores, recebido {len(vals)}")
        return vals

    def _on_apply_inputs(self):
        if self._forward_started:
            QMessageBox.information(self, 'Info', 'Reset antes de aplicar novos inputs.')
            return
        try:
            vals = self._parse_inputs_text()
            self.scene.update_input_values(vals)
            self.label_status.setText('Inputs prontos')
        except Exception as e:
            QMessageBox.warning(self, 'Erro', str(e))

    def _on_change_activation(self, name: str):
        if self._forward_started:
            QMessageBox.information(self, 'Info', 'Reset para mudar ativação.')
            # reverter seleção para atual
            idx = self.combo_activation.findText(self.network.activation_name)
            self.combo_activation.blockSignals(True)
            self.combo_activation.setCurrentIndex(idx)
            self.combo_activation.blockSignals(False)
            return
        try:
            self.network.set_activation(name)
            self.label_status.setText(f'Ativação: {name}')
        except Exception as e:
            QMessageBox.warning(self, 'Erro', str(e))

    def _update_node_colors(self):
        # mapa id->output
        outputs = {n.id: n.output for layer in self.network.layers for n in layer}
        # função cor baseada em output
        def out_color(v: float) -> QColor:
            if abs(v) < 1e-9:
                return QColor('#dddddd')
            # azul para positivo, vermelho para negativo
            ratio = min(1.0, abs(v))
            if v >= 0:
                r = int(70 - 50 * ratio)
                g = int(160 - 80 * ratio)
                b = int(240 - 120 * ratio)
            else:
                r = int(220 - 80 * (1 - ratio))
                g = int(70 - 50 * (1 - ratio))
                b = int(70 - 50 * (1 - ratio))
            return QColor(r, g, b)
        for item in self.scene.node_items:
            vid = getattr(item, 'node_id', None)
            if vid in outputs:
                base_pen = item.pen()
                item._base_brush = QBrush(out_color(outputs[vid]))
                if not item._highlight:
                    item.setBrush(item._base_brush)

    # ================= Treino ====================
    def _ensure_sample_for_training(self):
        """Garantir que temos (inputs, target) para treino.
        Se um sample já foi selecionado via dataset (self._current_target definido após _on_next_sample), usa-lo;
        Caso contrário, se dataset carregado, pega próximo sample;
        Se não houver dataset, tenta usar inputs customizados (sem target -> não treina) e aborta.
        """
        if self._dataset_loaded:
            if self._current_target is None:
                x, y = self.network.next_dataset_sample()
                self._current_target = y
                if not self.edit_inputs.text().strip():
                    self.edit_inputs.setText(', '.join(f'{v:.3f}' for v in x))
                self.scene.update_input_values(x)
                return x, y
            else:
                # inputs atuais a partir da UI
                try:
                    x = self._parse_inputs_text()
                except Exception:
                    # se falhar parse, re-obter sample
                    x, y = self.network.next_dataset_sample()
                    self._current_target = y
                    self.edit_inputs.setText(', '.join(f'{v:.3f}' for v in x))
                    self.scene.update_input_values(x)
                    return x, y
                return x, self._current_target
        else:
            # sem dataset
            if not self.edit_inputs.text().strip():
                QMessageBox.information(self, 'Info', 'Forneça inputs ou carregue dataset para treinar.')
                return None
            try:
                x = self._parse_inputs_text()
            except Exception as e:
                QMessageBox.warning(self, 'Erro', f'Inputs inválidos: {e}')
                return None
            QMessageBox.information(self, 'Info', 'Sem target (dataset). Não é possível calcular erro/backprop.')
            return None

    def _after_weight_update_refresh(self):
        # reconstruir cena para atualizar labels de pesos / setas / biases
        self.scene.build(self.network)
        # recolor outputs conforme estado atual
        self._update_node_colors()
        # se houver gradientes armazenados, aplicar overlay visual temporário
        self._apply_gradient_overlay()

    def _apply_gradient_overlay(self):
        grads = getattr(self.network, 'last_gradients', None)
        if not grads:
            return
        # normalizar magnitude
        mags = [abs(g) for g in grads.values()]
        max_mag = max(mags) if mags else 0.0
        if max_mag <= 0:
            return
        for edge_item in getattr(self.scene, 'edge_items', []):
            gid = getattr(edge_item, 'edge_id', None)
            if gid not in grads:
                continue
            g = grads[gid]
            ratio = min(1.0, abs(g) / max_mag) if max_mag else 0.0
            # cor: verde para grad positivo, magenta para negativo
            if g >= 0:
                r = int(40 + 40 * (1 - ratio))
                gcol = int(180 + 60 * ratio)
                b = int(60 + 40 * (1 - ratio))
            else:
                r = int(200 * ratio + 100 * (1 - ratio))
                gcol = int(50 * (1 - ratio) + 30)
                b = int(180 * ratio + 80 * (1 - ratio))
            pen = edge_item.pen()
            pen.setColor(QColor(r, gcol, b))
            pen.setWidthF(pen.widthF() + 1.5)
            edge_item.setPen(pen)
            edge_item.arrow_item.setBrush(QBrush(pen.color()))
        # timer para restaurar após curto intervalo
        QTimer.singleShot(1200, self._refresh_after_gradients)

    def _refresh_after_gradients(self):
        # reconstruir cena para voltar às cores base de peso (já com pesos atualizados)
        self.scene.build(self.network)
        self._update_node_colors()

    def _on_train_step(self):
        if self._forward_started:
            QMessageBox.information(self, 'Info', 'Use Reset antes de treinar (forward em andamento).')
            return
        sample = self._ensure_sample_for_training()
        if sample is None:
            return
        x, target = sample
        lr = float(self.spin_lr.value())
        outputs, error, loss = self.network.train_step(x, target, lr)
        self._record_loss(loss)
        self._after_weight_update_refresh()
        # atualizar labels de entrada conforme inputs efetivamente usados (já transformados internamente)
        self.scene.update_input_values(self.network.transform_inputs(x))
        self.label_status.setText(f'TrainStep out={outputs[0]:.4f} target={target:.4f} erro={error:.4f} loss={loss:.5f}')

    def _on_train_epoch(self):
        if not self._dataset_loaded:
            QMessageBox.information(self, 'Info', 'Carregue o dataset antes.')
            return
        if self._forward_started:
            QMessageBox.information(self, 'Info', 'Use Reset antes de treinar (forward em andamento).')
            return
        lr = float(self.spin_lr.value())
        total_loss = 0.0
        correct = 0
        total = len(self.network.dataset_inputs)
        for x, target in zip(self.network.dataset_inputs, self.network.dataset_targets):
            outputs, error, loss = self.network.train_step(x, target, lr)
            total_loss += loss
            pred = outputs[0]
            if (pred >= 0.5 and target >= 0.5) or (pred < 0.5 and target < 0.5):
                correct += 1
            self._record_loss(loss)
        # atualizar labels de entrada com o último x transformado
        if self.network.dataset_inputs:
            self.scene.update_input_values(self.network.transform_inputs(self.network.dataset_inputs[-1]))
        avg_loss = total_loss / total if total else 0.0
        acc = correct / total if total else 0.0
        self._after_weight_update_refresh()
        self.label_status.setText(f'Época: loss_med={avg_loss:.5f} acc={acc:.3f}')

    def _on_change_prep(self, name: str):
        if self._forward_started:
            QMessageBox.information(self, 'Info', 'Reset para mudar pré-processamento.')
            idx = self.combo_prep.findText(getattr(self.network, 'preprocess_mode', 'none'))
            self.combo_prep.blockSignals(True)
            if idx >= 0:
                self.combo_prep.setCurrentIndex(idx)
            self.combo_prep.blockSignals(False)
            return
        try:
            self.network.set_preprocess_mode(name)
            self.label_status.setText(f'Pré-processamento: {name}')
            # atualizar labels de entrada se houver texto atual
            if self.edit_inputs.text().strip():
                try:
                    raw = self._parse_inputs_text()
                    self.scene.update_input_values(self.network.transform_inputs(raw))
                except Exception:
                    pass
        except Exception as e:
            QMessageBox.warning(self, 'Erro', str(e))

    def _on_save_weights(self):
        try:
            self.network.save_json('weights.json')
            QMessageBox.information(self, 'Info', 'Pesos salvos em weights.json')
        except Exception as e:
            QMessageBox.warning(self, 'Erro', f'Falha ao salvar: {e}')

    def _on_load_weights(self):
        try:
            self.network.load_json('weights.json')
            self._refresh()
            QMessageBox.information(self, 'Info', 'Pesos carregados de weights.json')
        except Exception as e:
            QMessageBox.warning(self, 'Erro', f'Falha ao carregar: {e}')

    # ===== Loss History Window =====
    def _ensure_loss_window(self):
        if self._loss_window is None:
            self._loss_window = QWidget()
            self._loss_window.setWindowTitle('Histórico de Loss (últimos 200)')
            v = QVBoxLayout(self._loss_window)
            self.loss_text = QPlainTextEdit(); self.loss_text.setReadOnly(True)
            v.addWidget(self.loss_text)
            self._loss_window.resize(300, 400)
        return self._loss_window

    def _record_loss(self, value: float):
        self._loss_history.append(value)
        if len(self._loss_history) > 200:
            self._loss_history = self._loss_history[-200:]
        if self._loss_window and self._loss_window.isVisible():
            self._update_loss_text()
        if hasattr(self, 'loss_chart') and self.loss_chart is not None:
            self.loss_chart.set_history(self._loss_history)

    def _update_loss_text(self):
        if not hasattr(self, 'loss_text'):
            return
        lines = [f"{i:03d}: {v:.6f}" for i, v in enumerate(self._loss_history[-200:])]
        self.loss_text.setPlainText('\n'.join(lines))

    def _on_show_loss(self):
        win = self._ensure_loss_window()
        self._update_loss_text()
        win.show()


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
