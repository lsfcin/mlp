# MLP (Rede Neural como Grafo)

Projeto didático para aulas introdutórias de Inteligência Artificial. A ideia central é representar uma rede neural multicamada (MLP) como um grafo explícito composto por nós (neurônios) e arestas (conexões), evitando o uso direto de matrizes ou bibliotecas externas. O foco é a clareza estrutural, não a performance.

## Objetivos Didáticos
- Visualizar mentalmente a rede como um grafo: camadas, neurônios e conexões.
- Facilitar a compreensão de como conexões surgem entre camadas adjacentes.
- Permitir a modificação dinâmica da arquitetura (adição/remoção de neurônios) para explorar impactos estruturais.
- Incentivar boas práticas de nomenclatura em vez de comentários extensos: o código busca ser autoexplicativo.

## Filosofia do Código
- Sem comentários dentro das classes: nomes de classes, métodos e variáveis devem transmitir intenção.
- Estrutura modular: cada conceito em seu próprio arquivo.
- Simplicidade antes de otimização.
- Primeiro foco: construção e inspeção da topologia; próxima fase (não implementada ainda) poderá incluir propagação direta, funções de ativação, erro e ajuste de pesos.

## Estrutura de Arquivos (atualizada)
```
neuron.py       -> Classe Neuron (bias, output, delta)
connection.py   -> Conexões direcionadas (peso)
layer.py        -> Agrupamento de neurônios
network.py      -> Forward passo-a-passo, treino (backprop), registro de gradientes
graph_utils.py  -> Geração de dados para visualização (posições, cores, espessuras)
qt_viewer.py    -> Interface PyQt6 interativa (visualização + execução + treino)
loader.py       -> Carregador CSV simples (usa última coluna como alvo)
main.py         -> Exemplo mínimo (modo não-gráfico)
rsc/heart.csv   -> Dataset de exemplo
```

## Classes
### Neuron
Representa um nó. Mantém id único, bias, valores temporários (input_sum, output, delta). Nesta fase não executa ativação, apenas estrutura.

### Connection
Liga dois neurônios (source -> target) e guarda um peso inicial aleatório.

### Layer
Contém uma lista de neurônios e oferece métodos para adicionar ou remover.

### Network
Cria as camadas conforme tamanhos fornecidos e gera automaticamente todas as conexões entre camadas adjacentes. Permite adicionar ou remover neurônios (exceto na camada de entrada para preservar a correspondência com as features). Fornece um resumo textual da topologia.

### Loader
Lê o CSV (assumindo primeira linha como cabeçalho) e expõe número de features e número de linhas. Aqui consideramos, de forma simplificada, que a última coluna é o alvo e o restante são features.

## Como Executar
Pré‑requisito: Python 3 instalado.

No diretório do projeto, execute:
```
python main.py
```
Saída esperada (exemplo):
```
dados carregados: features=13 linhas=1025
rede criada: layers:13, 1 | connections:13
```
Isso indica que:
- 13 neurônios na camada de entrada (um por feature)
- 1 neurônio na camada de saída
- 13 conexões (cada neurônio de entrada ligado ao de saída)

## Interface Gráfica PyQt6
A interface (`qt_viewer.py`) abre uma janela com:
- Visualização da rede como grafo (nós = neurônios, arestas = pesos) com cores: azul (valores/pesos positivos), vermelho (negativos).
- Labels de bias em cada neurônio, pesos nas arestas, valores de entrada à esquerda, nomes de features acima, saída à direita.
- Zoom (scroll) e pan (arrastar). Highlight de neurônio atual no modo passo-a-passo.

### Controles Laterais
1. Camadas / Neurônios: adicionar ou remover camadas internas e neurônios (exceto na camada de entrada ou de saída para camadas).*  
2. Entrada: definir manualmente valores (separados por vírgula) antes de um forward.
3. Ativação: escolher `identity`, `sigmoid` ou `relu` (requer reset para mudar se já iniciou forward).
4. Dataset: carregar `heart.csv`, selecionar próximo sample, executar fast-forward em todas as amostras (mostra acurácia simples com threshold 0.5).
5. Execução: `▶` (play automático passo-a-passo), `Passo`, `⏸`, `Reset`.
6. Treino:
	- Campo LR (learning rate).
	- `Train Step`: faz um passo (forward + backprop) no sample atual (ou pega um sample do dataset se não houver seleção manual).
	- `Train Época`: percorre todo o dataset acumulando loss médio e acurácia.
	- `Ver Loss`: abre uma janela com histórico textual dos últimos 200 valores de loss.

### Visualização de Gradientes
Após qualquer passo de treino, as arestas são temporariamente recoloridas para representar o gradiente (dL/dw) da última atualização:
- Verde: gradiente positivo
- Magenta: gradiente negativo
A intensidade da cor e espessura aumentada refletem a magnitude relativa (normalizada). Após ~1.2s a rede é redesenhada com as cores base atualizadas pelos novos pesos.

### Forward Pass Passo-a-Passo
O forward pode ser acompanhado neurônio a neurônio. O status mostra o valor de saída de cada neurônio processado e, ao final, a saída e (se disponível) o erro em relação ao alvo.

### Treinamento
O treinamento usa MSE (loss = 0.5 * (o - t)^2) com backpropagation para um único neurônio de saída. Cada `Train Step`:
1. Executa forward completo.
2. Calcula deltas (gradientes internos).
3. Armazena gradientes por conexão (`network.last_gradients`).
4. Atualiza pesos e biases.
5. Atualiza visual e registra loss.

`Train Época` repete este fluxo para todas as amostras do dataset.

## Execução da Interface
Instale dependências (apenas PyQt6 neste estágio):
```
pip install PyQt6
```
Depois execute:
```
python qt_viewer.py
```

## Limitacões Atuais
- Saída única (backprop só cobre 1 neurônio de saída).
- Sem normalização de dados.
- Sem salvamento de pesos/estado.
- Visualização de loss é textual (não gráfico de linhas ainda).

## Próximos Passos (Sugestões)
- Gráfico de loss em tempo real (matplotlib embutido ou QPainter customizado).
- Suporte a múltiplos neurônios de saída / softmax.
- Persistência (salvar/carregar pesos em JSON ou pickle).
- Normalização / padronização de features.
- Mini-batches e baralhamento de dados.
- Exportar imagem da topologia / estados.

## Por Que Não Usar Matrizes Ainda?
A representação explícita facilita a introdução de conceitos fundamentais sem exigir abstrações lineares mais compactas. Após assimilação conceitual, a transição para operações vetoriais (NumPy, etc.) torna-se mais natural.

## Licença
Uso educacional. Adapte livremente para fins de ensino.

---
Sinta-se livre para estender a partir daqui. Para continuar, uma boa próxima etapa é adicionar um método `forward` na `Network` que consome um vetor de entradas e propaga valores camada a camada.
