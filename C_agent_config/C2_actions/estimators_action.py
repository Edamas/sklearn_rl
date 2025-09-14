import streamlit as st
import pandas as pd
from .utils import convert_column_to_list, convert_inf_values

def show_estimators():
    st.header("Estimadores do Agente")
    st.markdown("""
    **Estimadores** são algoritmos de machine learning que implementam métodos para:
    - **Aprendizado supervisionado**: Classificação e Regressão
    - **Aprendizado não supervisionado**: Clustering e Redução de dimensionalidade
    - **Pré-processamento**: Transformação e seleção de features
    
    Cada estimador possui parâmetros configuráveis que influenciam seu comportamento e performance.
    """)
    metadata_tab, view_tab = st.tabs(['Metadados', 'Exibição'])
    
    with metadata_tab:
        # Descrição resumida
        
        st.subheader("Descrição dos campos de `estimators.tsv`")
        st.markdown("""
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
        """)
    with view_tab:
        st.subheader("Tabela `estimators.tsv`")

        df = pd.read_csv(st.session_state.files.get('estimators'), sep='\t')
        
        # Inicialização da configuração das colunas
        column_config = {
            'exemplo_text': st.column_config.TextColumn('Exemplo', width='large'),
        }
        
        # conversão de colunas para list
        colunas_string_para_list = ['class_path', 'params_list', 'submethods_list', 'compatible_scores', 'output_y_types']
        for column in colunas_string_para_list:
            df[column] = convert_column_to_list(df[column])
            column_config[column] = st.column_config.ListColumn(column, width='medium')
        
        st.dataframe(
            df,
            column_config=column_config,
            width='stretch'
        )

