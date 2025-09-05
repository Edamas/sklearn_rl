# Documentação Completa da Aplicação: Análise e Visualização de Modelos Sklearn

Este documento oferece uma visão completa da aplicação, desde sua estrutura de alto nível até os detalhes de implementação de cada componente, seguindo a organização da interface.

## 1. Visão Geral e Propósito do Projeto

O projeto, intitulado **"Análise de Desempenho de Agente de IA Autônomo (AutoML + RL)"**, tem como objetivo principal fornecer uma interface interativa para analisar, visualizar e testar pipelines de modelos de aprendizado de máquina da suíte Scikit-learn.

A aplicação permite que o usuário carregue datasets, explore modelos compatíveis de forma visual e, futuramente, integre um agente de Aprendizado por Reforço (RL) para automatizar a construção de pipelines ótimos.

**Arquivo Principal:** `app.py` - Este arquivo é o ponto de entrada da aplicação Streamlit. Ele configura a página, define a estrutura de navegação e gerencia a exibição das diferentes seções.

---

## 2. Sumário Interativo da Aplicação

A tabela a seguir funciona como um sumário interativo. Clique nos links da coluna "Seção Principal" para navegar diretamente para a descrição detalhada de cada componente.

| Seção Principal                                             | Sub-seção(ões)                                    | Arquivo(s) Principal(is)                                                              | Função(ões) Chave                                        | Descrição Breve                                                                            |
| :---------------------------------------------------------- | :------------------------------------------------ | :------------------------------------------------------------------------------------ | :------------------------------------------------------- | :----------------------------------------------------------------------------------------- |
| [**Datasets**](#31-datasets-a-base-da-análise)             | Datasets Scikit-Learn                             | `datasets_app.py`<br>`data/`                                                          | `datasets_scikit_learn()`                                | Apresenta os conjuntos de dados disponíveis para análise.                                  |
| [**Agente RL (Conceito)**](#32-o-agente-de-rl-a-inteligência-autônoma-concepção) | Propósito, Ambiente, Estratégia, etc.             | `src/agent/`<br>`src/environment/`<br>`src/analysis/`                                  | `PPO`, `scikitlearn_env`                                 | Descreve a visão e a arquitetura do agente autônomo que fundamenta o projeto.            |
| [**Gráfico**](#33-gráfico-o-coração-da-análise-interativa) | Elaboração do Gráfico                             | `graph.py`                                                                            | `plot_sklearn_graph()`                                   | Interface principal para construção visual e interativa de pipelines de ML.                |
| [**Scikit-Learn**](#34-scikit-learn-o-catálogo-de-ferramentas) | Métodos Scikit-learn<br>Categorias de Métodos     | `sklearn_methods_app.py`<br>`sklearn_methods.tsv`<br>`categorias_sklearn.tsv`         | `show_sklearn_methods()`<br>`show_sklearn_categories()` | Catálogo de referência para métodos e categorias do Scikit-learn disponíveis no projeto. |
| **Documentação**                                            | README, Anotações, Cronograma, etc.               | `docs.py`<br>`docs/`                                                                   | `show_readme_md()`, etc.                                 | Reúne todos os documentos de apoio e planejamento do projeto.                              |

---

## 3. Detalhamento das Funcionalidades

### 3.1. Datasets: A Base da Análise

Esta seção, controlada pelo arquivo `datasets_app.py`, oferece uma visão geral dos dados disponíveis.

-   **Funcionamento:** A função `datasets_scikit_learn()` lê os metadados do arquivo `data/datasets_metadata.tsv` e os exibe em um formato claro. Para cada dataset, o usuário pode optar por visualizar as primeiras linhas do arquivo `.csv` correspondente, que está armazenado no diretório `data/`.

### 3.2. O Agente de RL: A Inteligência Autônoma (Concepção)

Embora a interface atual seja interativa e manual, ela foi projetada sobre um conceito mais amplo de um **agente autônomo de Aprendizado por Reforço (RL)**. As seções a seguir, extraídas do documento de concepção original, detalham essa visão.

#### 3.2.1. Propósito e Visão do Agente

O agente de RL é projetado para navegar pelo ecossistema Scikit-learn de forma autônoma. Seu objetivo é **aprender a construir e otimizar pipelines de Machine Learning** por meio de tentativa e erro, maximizando uma métrica de desempenho (recompensa) para um dado dataset.

**Arquivos Relevantes:**

-   `src/agent/`: Contém a implementação do agente (ex: `ppo_agent.py`).
-   `src/environment/`: Define o ambiente onde o agente opera (ex: `scikitlearn_env.py`).

#### 3.2.2. Ambiente, Ações e Recompensas

-   **Ambiente (`scikitlearn_env.py`):** É uma abstração sobre o Scikit-learn. O **estado** (observação) inclui metadados do dataset e o pipeline atual. O **espaço de ação** consiste em selecionar um método do `sklearn_methods.tsv`.
-   **Estratégia (PPO):** O agente usa o algoritmo *Proximal Policy Optimization* (PPO), implementado em `ppo_agent.py`, para aprender uma política que mapeia estados a ações.
-   **Sistema de Recompensa:** O agente é guiado por um sistema de recompensas e penalidades. A recompensa principal é proporcional ao score de desempenho (ex: F1-score) do pipeline final, enquanto penalidades são aplicadas para ações inválidas ou pipelines ineficientes.

#### 3.2.3. Plano de Implementação e Análise

1.  **Implementação:** O código-fonte do agente e do ambiente está localizado em `src/`.
2.  **Treinamento:** O arquivo `src/analysis/run_experiments.py` orquestra o ciclo de treinamento, onde o agente interage com o ambiente para aprender.
3.  **Análise de Resultados:** O arquivo `src/analysis/plot_results.py` é usado para visualizar a curva de aprendizado e analisar o desempenho do agente treinado.

#### 3.2.4. Proposta de Visualização Simbólica do Agente

O documento de concepção original propõe uma visualização do aprendizado do agente em um "mapa" 2D, onde o eixo X representa os passos do pipeline e o eixo Y as categorias de ações. A posição, cor e forma de um marcador representariam o estado e o sucesso das ações do agente em tempo real. A interface do "Gráfico" é um passo inicial e interativo em direção a essa visualização mais complexa e automatizada.

### 3.3. Gráfico: O Coração da Análise Interativa

Esta seção, implementada em `graph.py`, é a mais complexa e representa o núcleo funcional do projeto. Ela permite a construção de um pipeline de forma visual e guiada.

#### **Funcionamento Passo a Passo:**

1.  **Seleção do Dataset:** O usuário inicia selecionando um dos datasets disponíveis no menu lateral.
2.  **Carregamento e Análise do Input:**
    -   A função `load_dataset()` em `graph.py` carrega o arquivo `.csv` correspondente do diretório `data/`.
    -   A função `get_input_summary()` analisa o dataset e exibe um resumo na primeira coluna da interface, contendo:
        -   Número de linhas e colunas.
        -   Tipos de dados presentes (ex: int, float).
        -   Verificação de valores nulos.
        -   Estatísticas descritivas (mínimo, máximo).
3.  **Filtragem Automática de Modelos (N1):**
    -   Com base nos tipos de dados do dataset, o sistema filtra os modelos compatíveis para o primeiro nível de processamento (N1).
    -   A função `filter_models_by_type()` consulta o arquivo `sklearn_methods.tsv` para encontrar os modelos cujo `input_type` corresponde aos dados de entrada.
    -   Apenas os modelos compatíveis são exibidos em um campo de seleção múltipla (`st.multiselect`).
4.  **Conexão e Filtragem para N2:**
    -   Após a seleção de um ou mais modelos em N1, o sistema determina os tipos de output gerados por eles.
    -   Esses tipos de output são usados para filtrar e exibir os modelos compatíveis para o segundo nível (N2), garantindo que a conexão `N1 -> N2` seja válida.
5.  **Seleção de N2 e Definição do Output:**
    -   O usuário seleciona os modelos para N2. O output final do pipeline é determinado pelos tipos de dados de saída dos modelos escolhidos em N2.

**Arquivos de Configuração:**

-   `sklearn_methods.tsv`: Funciona como um banco de dados para os modelos. Contém colunas essenciais como `METHOD_NAME`, `input_type` e `output_type`, que são cruciais para a lógica de filtragem automática.
-   `graph.py`: Orquestra toda a lógica descrita acima, desde o carregamento dos dados até a renderização da interface em colunas no Streamlit.

### 3.4. Scikit-Learn: O Catálogo de Ferramentas

Esta área da aplicação serve como uma referência rápida sobre os componentes do Scikit-learn disponíveis no projeto.

-   **Funcionamento:**
    -   A página "Métodos Scikit-learn", gerenciada pela função `show_sklearn_methods()` em `sklearn_methods_app.py`, lê e exibe o conteúdo completo do arquivo `sklearn_methods.tsv` em uma tabela.
    -   A página "Categorias de Métodos", por sua vez, utiliza a função `show_sklearn_categories()` para exibir o arquivo `categorias_sklearn.tsv`, que agrupa os métodos em categorias funcionais (ex: Classificação, Regressão, Clusterização).