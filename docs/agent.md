# Documento de Concepção do Agente Autônomo

Este documento detalha a concepção, estratégia e plano de implementação para o agente de Inteligência Artificial Autônomo focado na utilização da suíte Scikit-learn. Ele evoluiu para ser uma ferramenta robusta de AutoML, capaz de construir e otimizar pipelines de Machine Learning de forma inteligente.

## 1. Visão Geral e Propósito do Agente

O agente é um sistema de Aprendizado por Reforço (RL) projetado para atuar de forma autônoma no ecossistema da biblioteca Scikit-learn. Seu propósito principal é aprender a construir e otimizar pipelines de Machine Learning (ML), desde o pré-processamento dos dados até a seleção e configuração do modelo final.

O objetivo é que o agente, por meio de tentativa e erro, descubra sequências de operações (ações) que maximizem uma métrica de desempenho (recompensa) para um determinado conjunto de dados (estado).

## 2. Modelo de Dados (Arquivos TSV)

A inteligência do agente é alimentada por três arquivos TSV principais, que servem como sua base de conhecimento e histórico.

### 2.1. `estimators.tsv`

Este arquivo contém o catálogo de todos os estimadores (modelos e transformadores) do Scikit-learn que o agente pode utilizar. Ele é a fonte primária de metadados sobre cada "caixa" do pipeline.

| Coluna | Tipo | Descrição | Exemplo |
| :--- | :--- | :--- | :--- |
| `estimator_name` | String | Nome do estimador (ex: `RandomForestClassifier`). | `PCA` |
| `estimator_type` | String | Tipo geral do estimador (`Classifier`, `Regressor`, `Transformer`, `Cluster`). | `Transformer` |
| `category` | String | Categoria de alto nível do estimador (ex: `ensemble`, `preprocessing`). | `decomposition` |
| `description` | String | Breve descrição do estimador, extraída da documentação. | `Principal component analysis (PCA).` |
| `class_path` | String | Caminho completo da classe Python do estimador. | `sklearn.decomposition.PCA` |
| `params_list` | Lista de Strings | Nomes dos parâmetros que o agente pode otimizar para este estimador. | `[n_components, whiten]` |
| `submethods_list` | Lista de Strings | Métodos públicos importantes do estimador (ex: `fit`, `transform`, `predict`). | `[fit, transform, inverse_transform]` |
| `X_min` | Inteiro | Número mínimo de features de entrada (`X`) que o estimador aceita. | `1` |
| `X_max` | Inteiro | Número máximo de features de entrada (`X`) que o estimador aceita. | `9999` |
| `y_min` | Inteiro | Número mínimo de features de saída (`y`) que o estimador aceita (para alvos). | `0` (para transformers) |
| `y_max` | Inteiro | Número máximo de features de saída (`y`) que o estimador aceita (para alvos). | `1` (para classificadores) |
| `apt_for_training` | Booleano | Indica se o estimador está pronto para ser usado no treinamento (`True`) ou se precisa de revisão (`False`). | `True` |
| `observações` | String | Notas e observações sobre o estimador (ex: inconsistências na documentação). | `Descrição fornecida incorreta.` |
| `from_sklearn_docs` | Booleano | Indica se o registro foi preenchido automaticamente a partir da documentação do Scikit-learn. | `True` |
| `input_X_structure` | String | Estrutura (shape) esperada para a entrada `X`. | `(n_samples, n_features)` |
| `input_X_types` | String | Tipos de dados aceitos para `X` (ex: `float,int`). | `float,int` |
| `input_y_structure` | String | Estrutura (shape) esperada para a entrada `y`. | `(n_samples,)` |
| `input_y_types` | String | Tipos de dados aceitos para `y` (ex: `float,int`). | `float,int` |
| `output_X_structure` | String | Estrutura (shape) da saída `X` após `transform`. | `(n_samples, n_components)` |
| `output_X_types` | String | Tipos de dados da saída `X` após `transform`. | `float` |
| `output_y_structure` | String | Estrutura (shape) da saída `y` após `predict`. | `(n_samples,)` |
| `output_y_types` | String | Tipos de dados da saída `y` após `predict`. | `float,int` |

### 2.2. `parameters.tsv`

Este arquivo detalha as regras de otimização para cada hiperparâmetro, permitindo que o agente gere valores válidos e explore o espaço de parâmetros de forma inteligente.

| Coluna | Tipo | Descrição | Exemplo |
| :--- | :--- | :--- | :--- |
| `param_name` | String | Nome do parâmetro (ex: `n_estimators`). | `n_components` |
| `param_dtype` | String | Tipo de dado do parâmetro (`int`, `float`, `cat` (categórico), `bool`). | `int` |
| `param_standard` | String | Valor padrão do parâmetro. | `None` |
| `param_min` | Float/Int | Valor mínimo para parâmetros numéricos. | `0.0` |
| `param_max` | Float/Int | Valor máximo para parâmetros numéricos. | `1.0` |
| `param_list` | Lista de Strings | Valores possíveis para parâmetros categóricos (ex: `['auto', 'full']`). | `['auto', 'full', 'arpack']` |
| `param_required` | Booleano | Indica se o parâmetro é obrigatório (`True`) ou opcional (`False`). | `False` |
| `descrição do parâmetro` | String | Breve descrição do parâmetro, extraída da documentação. | `Number of components to keep.` |
| `apt_for_training` | Booleano | Indica se o parâmetro está pronto para ser otimizado (`True`) ou se precisa de revisão (`False`). | `True` |
| `observações` | String | Notas e observações sobre o parâmetro. | `Pode ser int, float ou 'mle'.` |
| `from_sklearn_docs` | Booleano | Indica se o registro foi preenchido automaticamente a partir da documentação do Scikit-learn. | `True` |

### 2.3. `history.tsv`

Este arquivo registra os resultados de cada experimento (tentativa de pipeline e otimização de parâmetros) realizado pelo agente, servindo como base para o aprendizado e análise de desempenho.

| Coluna | Tipo | Descrição |
| :--- | :--- | :--- |
| `timestamp` | String | Data e hora da execução do experimento. |
| `duration_seconds` | Float | Tempo de execução do experimento em segundos. |
| `dataset_name` | String | Nome do dataset utilizado. |
| `dataset_summary` | String | Resumo do dataset (ex: número de features, amostras). |
| `estimator_name` | String | Nome do estimador principal utilizado. |
| `accuracy` | Float | Acurácia do modelo (para tarefas de classificação). |
| `r2_score` | Float | Coeficiente R² do modelo (para tarefas de regressão). |
| `error` | String | Mensagem de erro, se houver. |
| `[param_name]` | Vários | Uma coluna para cada parâmetro otimizado, com o valor utilizado. |

## 3. Fluxo de Trabalho do Agente

O agente opera em um ciclo contínuo de exploração e otimização, guiado pelos dados dos arquivos TSV.

### 3.1. Carregamento e Preparação de Dados

Ao iniciar, o agente carrega os arquivos `estimators.tsv` e `parameters.tsv` para sua memória, criando DataFrames Pandas que servem como sua base de conhecimento.

### 3.2. Filtragem de Estimadores Compatíveis

Antes de iniciar a otimização, o agente filtra os estimadores disponíveis com base na compatibilidade com o dataset atual e nas flags de `apt_for_training`.

| Critério de Filtragem | Descrição |
| :--- | :--- |
| `X_min`, `X_max`, `y_min`, `y_max` | Garante que o número de features e alvos do dataset esteja dentro do intervalo aceito pelo estimador. |
| `apt_for_training` | Apenas estimadores marcados como `True` são considerados para otimização. |
| `input_X_structure`, `input_X_types` | Verifica se a estrutura e os tipos de dados da entrada `X` do dataset são compatíveis com o que o estimador espera. |
| `input_y_structure`, `input_y_types` | Verifica se a estrutura e os tipos de dados da entrada `y` do dataset são compatíveis com o que o estimador espera. |

### 3.3. Geração de Hiperparâmetros

Para cada estimador compatível, o agente gera combinações aleatórias de hiperparâmetros. Esta geração é guiada pelas regras definidas no `parameters.tsv`.

| Regra de Geração | Descrição |
| :--- | :--- |
| `param_dtype` | Define se o valor gerado deve ser `int`, `float`, `cat` ou `bool`. |
| `param_min`, `param_max` | Para tipos numéricos, define o intervalo de valores. |
| `param_list` | Para tipos categóricos, define a lista de valores possíveis. |
| `param_standard` | Usado como valor padrão ou centro para a geração aleatória. |
| `apt_for_training` | Apenas parâmetros marcados como `True` são considerados para otimização. |

### 3.4. Treinamento e Avaliação do Modelo

O agente constrói um pipeline (com `StandardScaler` e o estimador selecionado), treina-o com os dados de treinamento e avalia seu desempenho usando validação cruzada (`cross_val_score`). O tempo de execução de cada experimento é registrado.

### 3.5. Registro de Histórico

Os resultados de cada experimento (incluindo os parâmetros utilizados, métricas de desempenho e tempo de execução) são registrados no `history.tsv`. Este histórico é fundamental para a análise de desempenho e para futuras otimizações.

### 3.6. Visualização de Resultados

A interface do Streamlit apresenta os resultados de forma clara, incluindo:

*   Tabelas dos resultados de todas as tentativas.
*   Gráficos de dispersão mostrando a performance de cada tentativa por modelo.
*   Um gráfico de dispersão comparando o score do modelo com o tempo de execução, permitindo uma análise de custo-benefício.

## 4. Estratégia de Compatibilidade de Pipeline

A capacidade do agente de construir pipelines complexos depende de um entendimento preciso da compatibilidade entre os estimadores. A analogia de "portas" é central aqui: a "porta de saída" de um estimador deve ser compatível com a "porta de entrada" do próximo.

### 4.1. Detalhamento dos Campos de Compatibilidade

As novas colunas em `estimators.tsv` fornecem essa informação detalhada:

| Campo | Descrição | Exemplo de Valor |
| :--- | :--- | :--- |
| `input_X_structure` | Descreve a dimensionalidade/shape esperada para a entrada `X` (ex: `(n_samples, n_features)`). | `(n_samples, n_features)` |
| `input_X_types` | Tipos de dados aceitos para `X` (ex: `float,int`). | `float,int` |
| `input_y_structure` | Estrutura (shape) esperada para a entrada `y`. | `(n_samples,)` |
| `input_y_types` | Tipos de dados aceitos para `y` (ex: `float,int`). | `float,int` |
| `output_X_structure` | Estrutura (shape) da saída `X` após `transform` (para transformers). | `(n_samples, n_components)` |
| `output_X_types` | Tipos de dados da saída `X` após `transform`. | `float` |
| `output_y_structure` | Estrutura (shape) da saída `y` após `predict`. | `(n_samples,)` |
| `output_y_types` | Tipos de dados da saída `y` após `predict`. | `float,int` |

### 4.2. Inferência a partir da Documentação

Esses campos são preenchidos por meio de uma análise cuidadosa da documentação do Scikit-learn. Minha lógica de processamento de arquivos de documentação (`docs/estimators_docs/*.txt`) é responsável por extrair essas informações das seções de `Parameters` e `Returns` dos métodos `fit`, `transform` e `predict`.

### 4.3. Regras de Conexão (Lógica do Agente)

A compatibilidade entre estimadores será determinada pela lógica do agente, que comparará as características de saída de um estimador com as características de entrada do próximo. Por exemplo:

*   Um estimador que produz `output_X_structure=(n_samples, n_features_new)` e `output_X_types=float` pode ser conectado a um estimador que aceita `input_X_structure=(n_samples, n_features)` e `input_X_types=float,int`.
*   Regras de conversão implícita (ex: `float` aceita `int`) serão incorporadas na lógica de compatibilidade do agente.

## 5. Representação de Contextos de Tamanho Variável para o Agente

Para que o agente possa tomar decisões eficazes, ele precisa processar informações de contextos que variam em tamanho, como datasets, o pipeline atual em construção e o histórico de experimentos. Abaixo, detalhamos as estratégias para converter esses inputs de tamanho variável em representações de tamanho fixo que o agente pode utilizar.

| Estratégia | Dataset | Pipeline Atual | Histórico de Experimentos |
| :--- | :--- | :--- | :--- |
| **Preenchimento/Truncamento** | Não ideal para datasets, pois a variação em `n_features` e `n_samples` é muito grande e heterogênea para um padding/truncamento simples sem perda de informação crítica. | **Representação:** Cada passo do pipeline (estimador + seus parâmetros) é codificado em um vetor de tamanho fixo. <br> **Aplicação:** Definir um `tamanho_max_pipeline` (ex: 10 passos). Se o pipeline atual tiver menos passos, preencher com vetores "vazios" (ex: zeros). Se tiver mais, truncar os passos mais antigos. <br> **Exemplo:** Para `tamanho_max_pipeline=5`, um pipeline `[vetor_Scaler, vetor_PCA, vetor_LogReg]` seria `[vetor_Scaler, vetor_PCA, vetor_LogReg, PAD_vec, PAD_vec]`. | **Representação:** Cada registro histórico (estado, ação, recompensa) é codificado em um vetor de tamanho fixo. <br> **Aplicação:** Definir um `tamanho_max_historico` (ex: 50 registros). Preencher/truncar a sequência de vetores históricos. <br> **Exemplo:** `[vetor_exp1, vetor_exp2, ..., vetor_expN, PAD_vec, ..., PAD_vec]` (onde N <= 50). |
| **Agregação/Sumarização** | **Representação:** Extrair um vetor fixo de estatísticas descritivas e metadados do dataset. <br> **Aplicação:** <br> • **Numéricas:** `n_features`, `n_samples`, média/std/min/max das features, estatísticas de correlação, proporção de nulos. <br> • **Categóricas:** Contagem de features categóricas, cardinalidade média, proporção de valores únicos. <br> • **Alvo:** Tipo (classificação/regressão), balanceamento de classes. <br> **Exemplo:** `[n_features, n_samples, avg_mean_X, avg_std_X, target_type_is_binary, target_type_is_regression, ...]` | Não é a estratégia ideal, pois perde a sequência e a ordem dos passos, que são cruciais para um pipeline. | **Representação:** Extrair um vetor fixo de estatísticas agregadas do histórico de experimentos. <br> **Aplicação:** Melhor score já atingido, score médio, desvio padrão dos scores, estimador mais frequente, duração média dos experimentos, número total de experimentos. <br> **Exemplo:** `[best_score_global, avg_score_last_100, std_score_all, most_freq_successful_estimator_ID, avg_duration_all, ...]` |
| **Redes Neurais Recorrentes (RNNs) / Transformers** | Não é o uso principal para o dataset como um todo. RNNs/Transformers são mais adequados para processar sequências *dentro* de um dataset, como séries temporais ou texto, não o dataset como uma única entidade. | **Representação:** A sequência de vetores de passos do pipeline é alimentada a uma RNN (ex: LSTM) ou a um Transformer **(tipicamente de frameworks de Deep Learning)**. <br> **Aplicação:** A rede processa a sequência passo a passo, mantendo um estado interno (memória) que resume o pipeline. O estado final (ou uma agregação das saídas) da rede é a representação de tamanho fixo do pipeline. <br> **Exemplo:** `RNN( [vetor_Scaler, vetor_PCA, vetor_LogReg] ) -> vetor_representacao_pipeline` | **Representação:** A sequência de vetores de registros históricos é alimentada a uma RNN ou a um Transformer **(tipicamente de frameworks de Deep Learning)**. <br> **Aplicação:** A rede aprende padrões temporais no histórico, como a evolução do desempenho ou a eficácia de certas estratégias. O estado final da rede é a representação de tamanho fixo do histórico. <br> **Exemplo:** `RNN( [vetor_exp1, vetor_exp2, ..., vetor_expN] ) -> vetor_representacao_historico` |
| **Redes Neurais Gráficas (GNNs)** | Não é o uso principal, a menos que o dataset tenha uma estrutura de grafo inerente, como redes sociais. | **Representação:** O pipeline é modelado como um grafo (nós = estimadores, arestas = fluxo de dados/conexões) **para ser processado por uma GNN (tipicamente de frameworks de Deep Learning)**. <br> **Aplicação:** Uma GNN processa a estrutura do grafo, aprendendo a agregar informações dos nós e arestas em uma representação de tamanho fixo. <br> **Exemplo:** `GNN( grafo_pipeline ) -> vetor_representacao_pipeline` | Não é o uso principal, a menos que o histórico seja uma sequência de grafos ou um grafo de interações. |

**Recomendação para Implementação Inicial:**

Para começar, sugiro focar na **Agregação/Sumarização** para a representação de **datasets** e **registros históricos**. Para a representação do **pipeline em construção**, podemos iniciar com **Preenchimento/Truncamento** se usarmos MLPs simples para o agente.

À medida que o agente evolui, podemos então explorar RNNs/Transformers para um tratamento mais sofisticado de dados sequenciais, e GNNs para estruturas de pipeline ainda mais complexas.

### Plano de Refatoração: Pré-processamento Automático e Entrada Padronizada para o Agente

Este plano detalha as mudanças necessárias para refatorar o sistema, permitindo que o agente de RL aprenda a construir pipelines de pré-processamento de forma dinâmica, com base em uma análise padronizada de qualquer dataset de entrada.

**Objetivo:** Abstrair a entrada do agente, passando de colunas individuais para grupos de colunas com características semelhantes. Isso permitirá que o agente generalize seu aprendizado para diferentes datasets e construa pipelines de pré-processamento mais robustos e adequados.

---

#### Fase 1: Análise e Agrupamento Automático de Colunas

1.  **Criar Nova Função de Análise (`functions.py`):**
    *   Desenvolver uma função `analyze_and_group_columns(df)`.
    *   **Lógica de Agrupamento:** Iterar sobre as colunas do dataset e classificá-las nos seguintes grupos com base em seu tipo e estatísticas:
        *   **Numéricas:** Colunas de tipo numérico (`int`, `float`).
        *   **Binárias:** Colunas numéricas com exatamente 2 valores únicos.
        *   **Categóricas (Texto):** Colunas de tipo `object` ou `string`, com um número de valores únicos relativamente baixo (ex: < 50).
        *   **Texto Livre:** Colunas de tipo `object` ou `string` com alta cardinalidade.
        *   **Datas:** Colunas que podem ser convertidas para o tipo `datetime`.
    *   **Lógica de Extração de Estatísticas:** Para cada grupo de colunas, calcular um conjunto fixo de estatísticas agregadas. Este será o vetor de features para o agente.
        *   **Para grupos Numéricos/Binários:** Média das médias, média dos desvios-padrão, média da contagem de nulos, etc.
        *   **Para grupos Categóricos/Texto:** Média da cardinalidade (valores únicos), média do comprimento das strings, etc.
        *   **Para grupos de Datas:** Intervalo (min/max), frequência média, etc.
    *   **Saída:** A função retornará um `DataFrame` de resumo onde cada linha representa um grupo de colunas, contendo as estatísticas e a lista de colunas pertencentes àquele grupo.

#### Fase 2: Adaptação da Interface do Usuário

1.  **Atualizar Seção de Datasets (`A_inputs/A1_datasets.py`):**
    *   Após o usuário selecionar um dataset, chamar a nova função `analyze_and_group_columns`.
    *   Dividir a seção "1. Seleção do Dataset" em duas:
        *   **1.1 Datasets Disponíveis:** A tabela de seleção de datasets existente.
        *   **1.2 Análise e Agrupamento de Atributos:** Exibir o novo DataFrame de resumo dos grupos de colunas (em modo somente leitura).
    *   Armazenar o DataFrame de resumo no `st.session_state` para ser usado como estado do agente.

2.  **Simplificar Definição de Features (`B_input_config/B1_features.py`):**
    *   Remover a tabela de edição de features.
    *   Substituí-la por um único `st.selectbox` para que o usuário identifique apenas a **coluna Alvo (Target `y`)**.
    *   Todas as outras colunas serão consideradas features (`X`) e já estarão organizadas nos grupos definidos na Fase 1.
    *   Determinar a tarefa (Classificação/Regressão) com base no tipo de dados da coluna alvo e salvar no `st.session_state`.

#### Fase 3: Refatoração do Agente e do Loop de Treinamento

1.  **Redefinir Estado e Ação do Agente (`D_training/agent_rl.py`):**
    *   **Estado (State):** O estado de entrada para o agente não será mais uma configuração manual, mas sim o **DataFrame de resumo dos grupos de colunas** (convertido para um vetor 1D). Este formato é fixo e padronizado.
    *   **Ação (Action):** A saída do agente (sua ação) será um conjunto de decisões de pré-processamento para cada *grupo* de colunas.
        *   *Exemplo de Ação para o grupo numérico:* `{ "imputer": "mean", "scaler": "standard" }`
        *   *Exemplo de Ação para o grupo categórico:* `{ "imputer": "most_frequent", "encoder": "onehot" }`

2.  **Construção Dinâmica de Pipeline (`D_training/D1_training.py`):**
    *   O centro da mudança estará na construção do pipeline do `scikit-learn`.
    *   Utilizar `sklearn.compose.ColumnTransformer` para aplicar diferentes sequências de transformações a diferentes grupos de colunas.
    *   **Fluxo no Loop de Treinamento:**
        1.  O agente recebe o estado (tabela de estatísticas) e gera as ações (escolhas de pré-processamento para cada grupo).
        2.  O código irá traduzir essas ações em instâncias de transformadores do `scikit-learn` (ex: `SimpleImputer`, `StandardScaler`, `OneHotEncoder`).
        3.  Um `ColumnTransformer` será montado dinamicamente, associando cada pipeline de grupo às suas respectivas colunas.
        4.  Este `preprocessor` será combinado com o estimador final em um `Pipeline` principal.
        5.  O pipeline completo será então avaliado usando `cross_val_score`.

#### Fase 4: Documentação

1.  **Atualizar `agent.md`:**
    *   Adicionar este plano detalhado ao final do arquivo `docs/agent.md` para registrar a nova arquitetura do agente.
