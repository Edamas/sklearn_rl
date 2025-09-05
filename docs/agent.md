# Documento de Concepção do Agente Autônomo

Este documento detalha a concepção, estratégia e plano de implementação para o agente de Inteligência Artificial Autônomo focado na utilização da suíte Scikit-learn.

## 1. Visão Geral e Propósito do Agente

O agente é um sistema de Aprendizado por Reforço (RL) projetado para atuar de forma autônoma no ecossistema da biblioteca Scikit-learn. Seu propósito principal é aprender a construir e otimizar pipelines de Machine Learning (ML), desde o pré-processamento dos dados até a seleção e configuração do modelo final.

O objetivo é que o agente, por meio de tentativa e erro, descubra sequências de operações (ações) que maximizem uma métrica de desempenho (recompensa) para um determinado conjunto de dados (estado).

## 2. Requisitos e Objetivos

### 2.1. Objetivo Principal

O objetivo central do agente é **maximizar a performance de um modelo de Machine Learning em um dado dataset**, automatizando a construção do pipeline de ponta a ponta. A performance será medida por métricas como acurácia, F1-score, precisão ou recall.

### 2.2. Requisitos Funcionais

- **Navegação no Espaço de Ações:** O agente deve ser capaz de selecionar qualquer estimador ou transformador disponível no Scikit-learn.
- **Construção de Pipeline Sequencial:** As ações do agente devem compor um `Pipeline` válido do Scikit-learn.
- **Avaliação de Desempenho:** O agente deve ser capaz de treinar e avaliar o pipeline construído em um conjunto de dados de validação.
- **Otimização da Recompensa:** O agente deve aprender uma política que o leve a escolher ações que resultem em uma maior recompensa acumulada ao longo do tempo.

## 3. Ambiente (Environment)

O ambiente representa o universo onde o agente opera. Neste projeto, o ambiente é uma abstração sobre a biblioteca Scikit-learn.

### 3.1. Espaço de Observação (Observation Space)

O estado (ou observação) representa a informação que o agente recebe do ambiente para tomar sua próxima decisão. O espaço de observação será composto por:

- **Metadados do Dataset:** Características do conjunto de dados (ex: número de features, número de amostras, tipo das features).
- **Estado do Pipeline Atual:** A sequência de passos de pré-processamento e modelagem já construídos.
- **Resultados Anteriores:** Informações sobre o desempenho de ações ou pipelines anteriores.

### 3.2. Espaço de Ação (Action Space)

O espaço de ação define todas as operações que o agente pode realizar. As ações consistem em:

- **Selecionar um método do Scikit-learn:** Escolher um algoritmo da lista de métodos disponíveis (ex: `StandardScaler`, `PCA`, `LogisticRegression`).
- **Configurar Hiperparâmetros:** Definir os parâmetros para o método escolhido (em uma versão futura).
- **Finalizar o Pipeline:** Indicar que a construção do pipeline está completa para que a avaliação possa começar.

## 4. Estratégia do Agente

### 4.1. Algoritmo de Aprendizagem

O agente utilizará um algoritmo de Aprendizado por Reforço. Com base na estrutura de arquivos do projeto (que sugere o uso de PPO - *Proximal Policy Optimization*), a estratégia será baseada em política (*policy-based*), onde o agente aprende diretamente a mapear estados para ações.

### 4.2. Sistema de Recompensa e Penalidades

O sistema de recompensa é crucial para guiar o aprendizado do agente na direção correta.

#### Recompensas

- **Recompensa Principal:** Proporcional à métrica de desempenho (ex: F1-score) do pipeline final no conjunto de validação. Um F1-score de 0.85 pode corresponder a uma recompensa de +85.
- **Recompensas Intermediárias:** Pequenos bônus por aplicar com sucesso um passo de pré-processamento que melhore a qualidade dos dados de alguma forma mensurável.

#### Penalidades

- **Ação Inválida:** Penalidade negativa significativa se o agente tentar uma ação que resulte em um pipeline inválido (ex: aplicar um classificador no meio de etapas de pré-processamento).
- **Custo Computacional:** Penalidade pequena e proporcional ao tempo de treinamento do pipeline, para incentivar a eficiência.
- **Desempenho Ruim:** Uma penalidade caso o desempenho final seja abaixo de um limiar mínimo aceitável.

## 5. Plano Operacional e Táticas

A implementação e o treinamento do agente seguirão as seguintes etapas:

1.  **Seleção dos Datasets:** Utilizar conjuntos de dados clássicos e bem conhecidos da comunidade (ex: Iris, Breast Cancer, Wine) para garantir a comparabilidade dos resultados.
2.  **Implementação do Ambiente (`scikitlearn_env.py`):**
    -   Desenvolver a classe de ambiente que herda de uma interface padrão (como a do `gymnasium`).
    -   Implementar a lógica de `step()`: receber uma ação, aplicá-la ao pipeline, calcular a recompensa e o próximo estado.
    -   Implementar a lógica de `reset()`: reiniciar o ambiente para o início de um novo episódio.
3.  **Implementação do Agente (`ppo_agent.py`):**
    -   Implementar o algoritmo PPO para interagir com o ambiente.
    -   Definir a arquitetura da rede neural que representará a política do agente.
4.  **Ciclo de Treinamento (`run_experiments.py`):**
    -   Orquestrar a interação entre o agente e o ambiente por um número definido de episódios.
    -   Salvar os resultados, as políticas aprendidas e as métricas de desempenho.
5.  **Análise e Avaliação (`plot_results.py`):**
    -   Visualizar a curva de aprendizado do agente (recompensa média por episódio).
    -   Analisar os pipelines gerados pelo agente treinado e compará-los com baselines manuais.

## 6. Estratégias e Exemplos Visuais

Para clarificar o funcionamento do agente, esta seção detalha algumas estratégias de alto nível que ele pode adotar e ilustra o ciclo de interação com o ambiente em cada caso.

### 6.1. Tabela de Estratégias Alternativas

O agente pode focar em diferentes aspectos da construção do pipeline. A tabela a seguir descreve algumas abordagens estratégicas.

| Estratégia Alternativa | Descrição |
| :--- | :--- |
| **Foco em Pré-processamento** | O agente prioriza a aplicação de uma sequência de transformadores para limpar, normalizar e preparar os dados antes de aplicar um modelo simples como baseline. |
| **Foco em Seleção de Modelo** | Dado um conjunto de dados já pré-processado, o agente se concentra em testar diferentes algoritmos de modelagem (classificação ou regressão) para encontrar o mais adequado. |
| **Construção de Pipeline Completo** | Estratégia mais complexa onde o agente constrói o pipeline de ponta a ponta, selecionando tanto os passos de pré-processamento quanto o modelo final. |

### 6.2. Exemplos de Ciclo (Entrada-Processamento-Saída)

#### Exemplo 1: Estratégia com Foco em Pré-processamento

| Processo | Entrada (Estado) | Processamento (Decisão do Agente) | Saída (Novo Estado + Recompensa) |
| :--- | :--- | :--- | :--- |
| **Passo 1: Imputação** | Dataset com 10% de valores faltantes; Pipeline vazio. | A política da rede neural analisa o estado e escolhe a ação: `SimpleImputer(strategy='mean')`. | Dataset sem valores faltantes; Pipeline: `[SimpleImputer]`; **Recompensa:** +5 (bônus por passo válido). |
| **Passo 2: Normalização** | Dataset sem valores faltantes; Pipeline: `[SimpleImputer]`. | A política escolhe a ação: `StandardScaler()`. | Dataset normalizado; Pipeline: `[SimpleImputer, StandardScaler]`; **Recompensa:** +5. |
| **Passo 3: Finalização** | Dataset normalizado; Pipeline: `[SimpleImputer, StandardScaler]`. | O agente aplica um modelo baseline (ex: `LogisticRegression`) e finaliza. | Pipeline avaliado; **Recompensa Final:** +78 (baseado no F1-score de 0.78 do pipeline completo). |

#### Exemplo 2: Estratégia com Foco em Seleção de Modelo

| Processo | Entrada (Estado) | Processamento (Decisão do Agente) | Saída (Novo Estado + Recompensa) |
| :--- | :--- | :--- | :--- |
| **Passo 1: Escolha do Modelo** | Dataset já pré-processado; Pipeline vazio. | A política avalia o estado e escolhe a ação: `KNeighborsClassifier(n_neighbors=5)`. | Pipeline finalizado com `KNeighborsClassifier`; **Recompensa:** +82 (baseado no F1-score). |
| **Episódio 2, Passo 1** | (Reset) Dataset pré-processado; Pipeline vazio. | Em um novo episódio, a política explora e escolhe: `RandomForestClassifier(n_estimators=100)`. | Pipeline finalizado com `RandomForestClassifier`; **Recompensa:** +91 (baseado no F1-score). |

## 7. Proposta de Visualização do Ambiente e Agente

Para fornecer uma visão intuitiva e em tempo real do processo de aprendizado do agente, propõe-se a criação de uma visualização simbólica utilizando Plotly. Este "mapa" não representa um espaço físico, mas sim o espaço abstrato de decisões do Scikit-learn.

### 7.1. O Mapa de Ações (O Ambiente)

O ambiente será visualizado como um gráfico 2D onde o agente se move da esquerda para a direita à medida que constrói o pipeline.

| Eixo | Representação |
| :--- | :--- |
| **Eixo X** | Etapas sequenciais do pipeline (Passo 1, Passo 2, ...). |
| **Eixo Y** | Categorias de ações do Scikit-learn (ex: Imputação, Normalização, Redução de Dimensionalidade, Classificação). |

Cada categoria no Eixo Y pode ter uma "região" colorida, e a trajetória do agente será uma linha que conecta os pontos de ação escolhidos em cada etapa.

### 7.2. O Agente e Seus Estados

O agente será um marcador no mapa, cujos atributos visuais mudam para refletir seu estado e o resultado de suas ações.

| Atributo Visual | Significado |
| :--- | :--- |
| **Posição (x, y)** | Representa a ação da **categoria Y** que foi tomada na **etapa X** do pipeline. |
| **Cor do Marcador** | Indica a fase atual do agente: <br> • **Azul:** Fase de Pré-processamento. <br> • **Verde:** Fase de Modelagem. <br> • **Dourado:** Episódio concluído com sucesso (alta recompensa). <br> • **Vermelho:** Episódio falhou (ação inválida). |
| **Forma do Marcador** | Descreve o resultado da última ação: <br> • **Círculo:** Ação válida, passo bem-sucedido. <br> • **'X' (xis):** Ação inválida que resultou em penalidade. |
| **Tamanho do Marcador** | Proporcional à magnitude da recompensa recebida no passo. Um marcador grande indica uma recompensa alta. |

### 7.3. Painel de Estatísticas

Um quadro de texto (anotação do Plotly) será posicionado próximo ao agente, exibindo estatísticas vitais em tempo real:

'''
--------------------
Episódio: 12
Passo: 4
Última Ação: RandomForestClassifier
Recompensa do Passo: +89.5
Recompensa Acumulada: 105.5
Melhor Score (Global): 0.92
--------------------
'''

### 7.4. Exemplo de Jornada Visual

1.  **Início:** O agente aparece na origem (0,0).
2.  **Passo 1:** O agente escolhe `SimpleImputer`. Ele se move para a posição `(1, 'Imputação')`. O marcador é um **círculo azul** de tamanho pequeno (recompensa intermediária).
3.  **Passo 2:** O agente escolhe `PCA`. Ele se move para `(2, 'Redução de Dim.')`. O marcador continua sendo um **círculo azul**.
4.  **Passo 3:** O agente escolhe uma ação inválida (ex: um segundo normalizador). Ele se move para `(3, 'Normalização')`, mas o marcador vira um **'X' vermelho** e pequeno (penalidade). O episódio termina.
5.  **Novo Episódio, Rota de Sucesso:** Em um novo episódio, após alguns passos de pré-processamento, o agente escolhe `RandomForestClassifier`. Ele se move para `(3, 'Classificação')`. O marcador vira um **círculo verde**.
6.  **Fim:** A avaliação é executada. O resultado é excelente. O marcador na posição final se torna um **círculo dourado** e grande, indicando o sucesso e a alta recompensa final.

## 8. Métricas de Avaliação e Recompensas por Etapa

Para guiar o aprendizado do agente de forma eficaz, o sistema de recompensas é estruturado com base em métricas avaliadas em diferentes estágios do pipeline. As tabelas a seguir detalham essa estrutura.

### 8.1. Etapa de Pré-processamento e Seleção de Features (A cada passo)

Nesta fase, o objetivo é incentivar a construção de uma sequência de transformações válida e útil.

| Métrica | O que Mede (Resultado Óbvio) | Recompensa (+) | Neutro (=) | Penalidade (-) |
| :--- | :--- | :--- | :--- | :--- |
| **Validade da Ação** | Se a operação é aplicável ao estado atual dos dados. | Ação aplicada com sucesso (pequeno bônus). | - | Erro de execução (`ValueError`, `TypeError`). |
| **Redução de Nulos** | Diminuição da porcentagem de valores faltantes. | A % de valores nulos diminuiu. | A % de nulos permaneceu a mesma. | - |
| **Redundância** | Se uma transformação idêntica ou funcionalmente similar já existe no pipeline. | - | Ação é nova e não redundante. | Aplicar a mesma classe de operação (ex: `StandardScaler` duas vezes). |
| **Dimensionalidade** | Mudança no número de features. | Redução de features com `PCA` ou `SelectKBest`. | - | Aumento inesperado ou remoção de todas as features. |

### 8.2. Etapa de Modelagem (No passo de seleção do modelo)

O foco aqui é garantir que o modelo escolhido seja apropriado para a tarefa.

| Métrica | O que Mede (Resultado Óbvio) | Recompensa (+) | Neutro (=) | Penalidade (-) |
| :--- | :--- | :--- | :--- | :--- |
| **Compatibilidade** | Se o modelo é adequado para o tipo de problema (Classificação vs. Regressão). | Modelo compatível com o target do dataset. | - | Modelo incompatível (resultando em erro). |
| **Finalização** | Se a escolha do modelo finaliza um pipeline válido. | Pipeline pronto para avaliação. | - | - |

### 8.3. Etapa de Avaliação Final (No fim de cada episódio)

Esta é a etapa mais importante, onde a qualidade geral do pipeline é julgada e a maior parte da recompensa é atribuída.

| Métrica | O que Mede (Resultado Óbvio) | Recompensa (+) | Neutro (=) | Penalidade (-) |
| :--- | :--- | :--- | :--- | :--- |
| **Score de Desempenho** | Métrica principal (ex: F1-Score, Acurácia) no conjunto de validação. | Score alto (ex: > 0.85). A recompensa é proporcional ao score. | Score mediano (ex: ~0.70). | Score baixo ou inaceitável (ex: < 0.5). |
| **Tempo de Execução** | Tempo total para treinar e avaliar o pipeline completo. | Tempo abaixo de um limiar de eficiência. | - | Tempo excessivamente longo, indicando um pipeline ineficiente. |
| **Complexidade** | Número de passos no pipeline final. | Pipeline enxuto e com bom desempenho. | - | Pipeline muito longo sem ganho de performance significativo. |

## 9. Catálogo de Tarefas do Agente

A interação do usuário com o agente é centrada no **propósito**. O usuário escolhe uma das tarefas de alto nível listadas abaixo, fornece os dados e seleciona uma métrica de otimização. O agente então assume a responsabilidade de explorar as diversas alternativas (modelos e pré-processadores) para encontrar a melhor solução para aquele propósito.

A tabela a seguir representa o "menu de serviços" do agente.

| Propósito do Usuário (O que você quer fazer?) | Alternativas (Métodos que o Agente irá Testar) | Métricas de Otimização Aplicáveis |
| :--- | :--- | :--- |
| **Classificar Dados**<br><small>Preciso classificar meus dados em duas ou mais categorias.</small> | `LogisticRegression`<br>`SVC`<br>`RandomForestClassifier`<br>`KNeighborsClassifier`<br>`GaussianNB`<br>`DecisionTreeClassifier`<br>`GradientBoostingClassifier` | `accuracy`<br>`f1_weighted`<br>`roc_auc`<br>`precision_weighted`<br>`recall_weighted` |
| **Prever um Valor Numérico (Regressão)**<br><small>Preciso prever um valor contínuo.</small> | `LinearRegression`<br>`SVR`<br>`RandomForestRegressor`<br>`KNeighborsRegressor`<br>`Lasso`<br>`Ridge`<br>`ElasticNet`<br>`GradientBoostingRegressor` | `r2`<br>`neg_mean_squared_error`<br>`neg_mean_absolute_error` |
| **Agrupar Dados (Clusterização)**<br><small>Preciso agrupar meus dados em grupos similares, sem uma variável alvo.</small> | `KMeans`<br>`DBSCAN`<br>`AgglomerativeClustering`<br>`Birch`<br>`MeanShift` | `silhouette_score`<br>`davies_bouldin_score`<br>`calinski_harabasz_score` |
| **Reduzir a Dimensionalidade**<br><small>Preciso reduzir o número de colunas (features) dos meus dados.</small> | `PCA`<br>`TSNE`<br>`FactorAnalysis`<br>`FastICA`<br>`SelectKBest` | *A recompensa é indireta:*<br>• Maximização da variância explicada.<br>• Minimização do erro de reconstrução.<br>• Melhora no score de um modelo downstream. |

**Nota sobre o Processo:** Ao escolher um propósito como "Classificar Dados", o agente não irá apenas testar os diferentes modelos de classificação. Ele também irá, de forma autônoma, explorar o espaço de ações de **pré-processamento** e **seleção de features** para construir o pipeline mais performático possível para cada modelo que ele testa. O usuário define o "o quê", e o agente otimiza o "como".

## 10. Resumo do Projeto: Análise e Visualização de Conexões

### Objetivo Geral:
O projeto tem como objetivo criar uma interface interativa em Streamlit para analisar, visualizar e testar pipelines de modelos de aprendizado de máquina da suíte Scikit-learn, usando datasets carregados pelo usuário. Ele integra filtragem automática de modelos compatíveis com base no tipo e nas características do dataset, permitindo que o usuário explore combinações de modelos de forma visual e controlada.

### Estrutura Geral

#### Dataset Automático

- O usuário seleciona um dataset no menu principal (st.session_state.selected_dataset).
- O dataset é carregado automaticamente de data/{nome_dataset}.csv.
- Informações sobre o dataset são exibidas de forma descritiva na primeira coluna:
  - Linhas e colunas
  - Tipos gerais de dados (int, float, object)
  - Presença de valores nulos
  - Valores mínimos e máximos de colunas numéricas

#### Interface de Conexão de Modelos

- A visualização é organizada em 7 colunas no Streamlit:
  - Input Dataset (dados do dataset)
  - Conexão Input → N1 (tipos compatíveis)
  - N1 - Modelos compatíveis
  - Conexão N1 → N2
  - N2 - Modelos compatíveis
  - Conexão N2 → Output
  - Output (determinado pelos modelos N2)
- Cada coluna de conexão mostra apenas os tipos de dados compatíveis com o dataset de entrada.
- As colunas de N1 e N2 exibem apenas os modelos compatíveis com os tipos de dados do dataset, permitindo seleção via multiselect.

#### Filtragem Automática

- O graph.py garante que os modelos disponíveis em N1 e N2 sejam compatíveis com o dataset carregado.
- Cada multiselect possui uma key única baseada no dataset para evitar conflitos no Streamlit.
- A filtragem é feita inicialmente pelo tipo de dados (int, float) das colunas do dataset.
- Futuramente, a filtragem pode incluir intervalos de valores mínimos e máximos para compatibilidade mais rigorosa.

#### Arquivos de Configuração

- **sklearn_methods.tsv:** contém todos os modelos disponíveis, com informações sobre tipos de input e output, usados para filtrar automaticamente os modelos compatíveis.
- **graph.py:** contém toda a lógica de carregamento do dataset, resumo do input, filtragem de modelos e exibição no Streamlit.

### Comportamento Dinâmico

Assim que o usuário seleciona um dataset, a interface:

- Exibe automaticamente o resumo do dataset
- Atualiza as conexões para N1 e N2
- Filtra os modelos compatíveis

Mensagens de aviso aparecem caso:

- Nenhum dataset tenha sido selecionado
- Nenhum modelo compatível esteja disponível
- Nenhum modelo N1 tenha sido selecionado antes de N2

### Próximos Passos e Funcionalidades Possíveis

- Filtragem avançada com base em valores mínimos e máximos reais das colunas.
- Conexão dinâmica entre N1 e N2, onde N2 exibe apenas os modelos compatíveis com os N1 selecionados.
- Visualização de pipelines completos de ML e seus outputs esperados.

### Resumo do graph.py:

#### Funções principais:

- **load_dataset():** carrega o dataset selecionado pelo usuário.
- **get_input_summary(df):** extrai informações gerais sobre o dataset.
- **load_methods():** carrega os métodos do sklearn_methods.tsv.
- **filter_models_by_type(df, allowed_types):** filtra os modelos de acordo com os tipos de input do dataset.
- **plot_sklearn_graph():** função principal que organiza a interface, mostra o input, conexões, modelos N1/N2 e output.

#### Lógica de visualização:

- Se o dataset não existir ou não estiver selecionado, exibe mensagem de alerta.
- O resumo do input é exibido na primeira coluna.
- Conexões entre modelos mostram tipos compatíveis.
- Multiselects permitem selecionar os modelos compatíveis, filtrados automaticamente pelo dataset.
- Output é definido a partir da seleção de N2.


# Estratégia de Agente para Otimização de Estimadores Scikit-learn

## 1. Tabela de Referência por Estimator

Cada estimator terá uma tabela de referência de parâmetros com os seguintes campos:

| Parâmetro | Tipo | Valor Padrão | Obrigatório | Intervalo / Valores Possíveis | Condicionalidade |
|-----------|------|--------------|-------------|-------------------------------|-----------------|
| param_name | int / float / bool / categorical | default | True / False | [min, max] ou lista de categorias | Ex.: só usado se outro_param=True |

**Exemplo:**  
| param_name       | type   | default   | required | range / values        | conditional          |
|-----------------|--------|----------|---------|----------------------|--------------------|
| n_estimators     | int    | 100      | False   | [10, 1000]           | -                  |
| learning_rate    | float  | 0.1      | False   | [0.01, 1.0]          | -                  |
| criterion        | str    | "gini"   | False   | ["gini", "entropy"]  | -                  |
| max_depth        | int    | None     | False   | [1, 50]              | -                  |

Essa tabela servirá como **referência para o agente**, permitindo saber os tipos, defaults, ranges e se os parâmetros são obrigatórios ou condicionais.

---

## 2. Inicialização do Agente

Para cada parâmetro do estimator selecionado:

1. Gerar valores aleatórios dentro do **range especificado** ou centrados no **valor default**.  
2. Utilizar um **desvio inicial grande** (`std`) para explorar amplamente o espaço de parâmetros.  
3. Registrar cada combinação gerada junto com o estimator utilizado.

Exemplo de inicialização aleatória:  

param_value = np.random.normal(loc=default_value, scale=initial_std)

## 3. Treinamento Iterativo

O agente executa ciclos de treinamento seguindo estas etapas:

1. **Seleção de Estimator e Geração de Parâmetros**
   - Escolhe um estimator.
   - Gera valores de parâmetros aleatórios com base na tabela de referência (types, defaults, ranges, required, conditional).

2. **Construção do Modelo**
   - Constrói o modelo usando: `estimator(**params)`.

3. **Treinamento**
   - Treina o modelo em `X_train`, `y_train`.

4. **Avaliação de Performance**
   - Avalia o desempenho em `X_test`, `y_test`.
   - Métricas possíveis: `score`, `loss`, `accuracy`, entre outras.

5. **Registro de Resultados**
   - Armazena cada experimento em uma tabela de histórico:
   - estimator param_combination resultado timestamp

6. **Atualização de Distribuições de Parâmetros**
   - Ajusta a média de cada parâmetro com base nos valores que geraram bons resultados.
   - Reduz progressivamente o desvio (`std`), refinando a exploração do espaço de parâmetros.

## 4. Geração do “Tabelão” de Parâmetros

Ao longo do treinamento, o agente constrói um tabelão com todas as combinações testadas, contendo:

- Estimator utilizado
- Valores de cada parâmetro
- Métricas de performance (score, loss, etc.)
- Número de iteração / timestamp

**Exemplo:**

| estimator                   | n_estimators | learning_rate | max_depth | score |
|------------------------------|--------------|---------------|-----------|-------|
| GradientBoostingClassifier    | 100          | 0.1           | 3         | 0.87  |
| GradientBoostingClassifier    | 150          | 0.05          | 5         | 0.89  |
| RandomForestClassifier        | 100          | -             | None      | 0.85  |
| RandomForestClassifier        | 200          | -             | 10        | 0.88  |
| HistGradientBoostingRegressor | 100          | 0.1           | 31        | 0.82  |
| HistGradientBoostingRegressor | 150          | 0.05          | 20        | 0.84  |
| StackingClassifier            | -            | -             | -         | 0.86  |
| GradientBoostingClassifier    | 120          | 0.08          | 4         | 0.90  |
| RandomForestRegressor         | 100          | -             | None      | 0.80  |
| HistGradientBoostingClassifier| 100          | 0.1           | 31        | 0.88  |

Este tabelão permite ao agente visualizar o impacto de cada parâmetro e ajustar futuras combinações de forma inteligente.

## 5. Aprendizado Multivariado

### 5.2. Geração de Novas Combinações

Com o modelo previsor ajustado, o agente pode:
- Predizer a performance de novas combinações de parâmetros antes de testá-las.
- Selecionar as combinações mais promissoras para reduzir exploração aleatória.
- Atualizar gradualmente a distribuição dos parâmetros:

``` python
# Exemplo simplificado
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# X: matriz de parâmetros já testados
# y: scores correspondentes
model = RandomForestRegressor()
model.fit(X, y)

# Gerar novas combinações aleatórias
new_params = np.random.uniform(low=param_min, high=param_max, size=(100, n_params))

# Predizer a performance
predicted_scores = model.predict(new_params)

# Selecionar top-k combinações
top_indices = np.argsort(predicted_scores)[-10:]
top_params = new_params[top_indices]
```
### 5.3. Reforço e Exploração Guiada

O agente aplica uma estratégia de exploração-exploração:

- Exploração ampla no início: usar desvio grande para testar muitos parâmetros diferentes.
- Exploração guiada: priorizar regiões com combinações que historicamente deram melhores resultados.
- Ajuste adaptativo: atualizar média e desvio (std) de cada parâmetro com base no desempenho observado.

Visualmente:
``` nginx
Score
^
|          ████
|     ████ ████
|  ████ ████ ████
|████ ████ ████ ████
+--------------------> Espaço de parâmetros
```
As barras indicam regiões do espaço de parâmetros onde a performance foi mais alta.

### 5.4. Benefícios dessa Estratégia

- Reduz o número de testes inúteis.
- Aprimora rapidamente a escolha de parâmetros.
- Captura relações não-lineares e interações complexas.
- Permite adaptação dinâmica conforme mais resultados são coletados.

## 6. Resumo da Estratégia

Consultar a tabela de referência para cada estimator e seus parâmetros.

Inicializar parâmetros aleatoriamente com base nos defaults e ranges.

Treinar iterativamente, avaliar performance e armazenar histórico.

Atualizar distribuição dos parâmetros (média e std) conforme aprende.

Criar um tabelão completo de experimentos para análise e refinamento.

Usar abordagem multivariada adaptativa para reduzir exploração e aumentar exploração guiada.