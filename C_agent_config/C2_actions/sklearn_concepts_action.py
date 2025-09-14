import streamlit as st

def show_sklearn_concepts():
    st.header("Conceitos Scikit-learn", divider='rainbow')

    st.markdown("""
    Esta seção fornece uma visão geral dos principais conceitos do Scikit-learn que o agente utiliza para construir pipelines de machine learning.
    """)

    tab_estimators_tsv, tab_params_tsv, tab_tutorial = st.tabs(["Tabela de Estimadores", "Tabela de Parâmetros", "Tutorial de Conceitos"])

    with tab_estimators_tsv:
        st.subheader("`estimators.tsv`")
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

    with tab_params_tsv:
        st.subheader("`parameters.tsv`")
        st.markdown("""
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
        """)

    with tab_tutorial:
        st.header("Tutorial de Conceitos", divider=True)
        tab_transformers, tab_estimators = st.tabs(['TRANSFORMERS', 'ESTIMADORES'])
    
        with tab_transformers:
            st.header('TRANSFORMERS', divider=True)
            st.subheader('Qualquer objeto que CONVERTE dados a partir de Dados')
            st.markdown('a) EXEMPLO DE TRANSFORMER: Redutor de dimensionalidade PCA - Principal Component Analysis')
            st.caption('Input')
            st.code('''
            X = [
            [4, 1, 2, 2],
            [1, 3, 9, 3],
            [5, 7, 5, 1]
            ]''')
            st.caption('Processing')
            st.code('''normalizer = sklearn.preprocessing.Normalizer()
            normalizer.fit(X)
            normalizer.transform(X)
            print("Saída transformada:\n", normalizer)''')
            st.caption('Output')
            st.code('''Saída transformada:
            [
            [0.8164 0.2041 0.4082 0.4082]
            [0.0890 0.2672 0.8017 0.2672]
            [0.4915 0.6881 0.4915 0.0983]
            ]''')
        
        with tab_estimators:
            st.header('ESTIMADORES', divider=True)
            st.subheader('Qualquer objeto que APRENDE a partir de Dados')
            
            # Criando subtabs para os tipos de estimadores
            tab_supervised, tab_unsupervised = st.tabs(['APRENDIZADO SUPERVISIONADO', 'APRENDIZADO NÃO-SUPERVISIONADO'])
            
            with tab_supervised:
                st.subheader('Aprendizado supervisionado prevê com dados rotulados pré-definidos')
                
                # Tab para Classificação
                tab_classification, tab_regression = st.tabs(['CLASSIFICAÇÃO', 'REGRESSÃO'])
                
                with tab_classification:
                    st.markdown('Classificação (Classifier) - RandomForestClassifier')
                    st.caption('Input')
                    st.code('''X = [[0, 0], [1, 1], [0, 1], [1, 0]]
                    y = [0, 1, 1, 0]   # rótulos''')
                    st.caption('Processing')
                    st.code('''classificador = RandomForestClassifier(random_state=0)
                    classificador.fit(X, y)

                    y_pred = classificador.predict([[0, 0], [1, 1], [0, 1]])''')
                    st.caption('Output')
                    st.code('''Predições: [0 1 1]''')
                
                with tab_regression:
                    st.markdown('Regressão (Regressor) - RandomForestRegressor')
                    st.caption('Input')
                    st.code('''X = [[0, 0], [1, 1], [2, 2], [3, 3]]
                    y = [0.0, 1.0, 2.0, 3.0]   # valores contínuos''')
                    st.caption('Processing')
                    st.code('''regressor = RandomForestRegressor(random_state=0)
                    regressor.fit(X, y)

                    y_pred = regressor.predict([[1.5, 1.5], [2.5, 2.5]])''')
                    st.caption('Output')
                    st.code('''Predições: [1.3 2.5]''')
            
            with tab_unsupervised:
                st.subheader('Aprendizado não-supervisionado')
                st.subheader('Aprendizado não-supervisionado prevê GRUPOS sem dados rotulados pré-definidos')
                
                # Tab para Clustering
                tab_clustering = st.tabs(['CLUSTERING'])[0]
                
                with tab_clustering:
                    st.markdown('Agrupamento (Clustering) – DBSCAN')
                    st.caption('Input')
                    st.code('''X = [[1, 2], [2, 2], [2, 3],
                        [8, 7], [8, 8], [25, 80]]''')
                    st.caption('Processing')
                    st.code('''clusters = DBSCAN(eps=3, min_samples=2).fit(X)''')
                    st.caption('Output')
                    st.code('''Labels atribuídos: [ 0  0  0  1  1 -1 ]
                    # -1 significa ponto considerado ruído (outlier)''')
