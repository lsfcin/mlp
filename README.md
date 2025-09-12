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

## Estrutura de Arquivos
```
neuron.py      -> Neuron: nó básico com bias e valores intermediários
connection.py  -> Connection: ligação direcionada com peso entre dois neurônios
layer.py       -> Layer: agrupamento de neurônios de mesma profundidade
network.py     -> Network: orquestra camadas e gera conexões entre elas
loader.py      -> Loader: simples carregador de dataset CSV (heart.csv)
main.py        -> Exemplo mínimo de uso
rsc/heart.csv  -> Dataset (13 features + alvo)
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

## Próximos Passos (Sugestões)
- Implementar forward pass com função de ativação configurável (sigmoid, ReLU, etc.).
- Incluir cálculo de erro e rascunho de backpropagation.
- Suporte a inicialização customizada de pesos e biases.
- Visualização (ex: exportar para formato DOT/Graphviz).
- Separar claramente features e alvo no Loader (atualmente inferido de forma simples).
- Persistir e recarregar pesos.

## Limitações Atuais
- Sem treinamento ou ajustes de pesos.
- Sem suporte a minibatches ou normalização.
- Loader assume CSV bem formatado (não há validações robustas).
- Não há testes automatizados ainda.

## Por Que Não Usar Matrizes Ainda?
A representação explícita facilita a introdução de conceitos fundamentais sem exigir abstrações lineares mais compactas. Após assimilação conceitual, a transição para operações vetoriais (NumPy, etc.) torna-se mais natural.

## Licença
Uso educacional. Adapte livremente para fins de ensino.

---
Sinta-se livre para estender a partir daqui. Para continuar, uma boa próxima etapa é adicionar um método `forward` na `Network` que consome um vetor de entradas e propaga valores camada a camada.
