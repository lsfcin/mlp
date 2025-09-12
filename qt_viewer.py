from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import (
    QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QGraphicsLineItem,
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSpinBox, QLabel, QFrame, QMessageBox
)
from PyQt6.QtGui import QBrush, QPen, QColor, QTransform, QPainter
from PyQt6.QtCore import Qt, QPointF

from network import Network
from graph_utils import graph_data, add_layer, remove_layer


class GraphNodeItem(QGraphicsEllipseItem):
    def __init__(self, node: Dict[str, Any]):
        size = node['width'] * 4
        super().__init__(-size/2, -size/2, size, size)
        self.node_id = node['id']
        self.setBrush(QBrush(QColor(node['color'])))
        pen = QPen(QColor(node['color']))
        pen.setWidthF(max(1.0, node['width'] * 0.5))
        self.setPen(pen)
        self.setToolTip(node['label'])
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setPos(QPointF(node['x'], node['y']))


class GraphEdgeItem(QGraphicsLineItem):
    def __init__(self, src: Dict[str, Any], dst: Dict[str, Any], edge: Dict[str, Any]):
        super().__init__(src['x'], src['y'], dst['x'], dst['y'])
        pen = QPen(QColor(edge['color']))
        pen.setWidthF(max(0.5, edge['width']))
        self.setPen(pen)
        self.setToolTip(edge['label'])
        self.setZValue(-1)


class GraphScene(QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.setBackgroundBrush(QBrush(QColor('#ffffff')))

    def build(self, network: Network):
        self.clear()
        data = graph_data(network)
        nodes_index = {n['id']: n for n in data['nodes']}
        # edges first so nodes appear on top
        for e in data['edges']:
            src = nodes_index[e['source']]
            dst = nodes_index[e['target']]
            self.addItem(GraphEdgeItem(src, dst, e))
        for n in data['nodes']:
            self.addItem(GraphNodeItem(n))
        self.setSceneRect(self.itemsBoundingRect().adjusted(-80, -80, 80, 80))


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
        btn_play = QPushButton('▶')
        btn_step = QPushButton('⏯ passo')
        btn_pause = QPushButton('⏸')
        btn_reset = QPushButton('Reset')
        for b in (btn_play, btn_step, btn_pause, btn_reset):
            h_play.addWidget(b)
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
