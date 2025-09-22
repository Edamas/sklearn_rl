import streamlit as st
import pandas as pd
from .utils import convert_column_to_list, convert_inf_values
from functions import create_optimized_estimators_tsv, show_data_by_function

def show_sklearn_concepts():
    st.header("Dashboard & Conceitos", divider='rainbow')

    st.markdown("""
    Esta seção fornece uma visão geral dos principais conceitos e artefatos do projeto.
    """)

    tab_conceitos, tab_estimators, tab_params, tab_cronograma, tab_registro = st.tabs([
        "Conceitos Scikit-learn", 
        "Tabela de Estimadores", 
        "Tabela de Parâmetros", 
        "Cronograma do Projeto", 
        "Registro de Atividades"
    ])

    with tab_conceitos:
        st.header("Tutorial de Conceitos", divider=True)
        tab_transformers, tab_estimators_concept = st.tabs(['TRANSFORMERS', 'ESTIMADORES'])
    
        with tab_transformers:
            st.header('TRANSFORMERS', divider=True)
            st.subheader('Qualquer objeto que CONVERTE dados a partir de Dados')
            st.markdown('a) EXEMPLO DE TRANSFORMER: Redutor de dimensionalidade PCA - Principal Component Analysis')
            st.caption('Input')
            st.code("""
            X = [
            [4, 1, 2, 2],
            [1, 3, 9, 3],
            [5, 7, 5, 1]
            ]
            """)
            st.caption('Processing')
            st.code("""normalizer = sklearn.preprocessing.Normalizer()
            normalizer.fit(X)
            normalizer.transform(X)
            print("Saída transformada:\n", normalizer)""", language='python')
            st.caption('Output')
            st.code("""Saída transformada:
            [
            [0.8164 0.2041 0.4082 0.4082]
            [0.0890 0.2672 0.8017 0.2672]
            [0.4915 0.6881 0.4915 0.0983]
            ]""", language='python')
        
        with tab_estimators_concept:
            st.header('ESTIMADORES', divider=True)
            st.subheader('Qualquer objeto que APRENDE a partir de Dados')
            
            tab_supervised, tab_unsupervised = st.tabs(['APRENDIZADO SUPERVISIONADO', 'APRENDIZADO NÃO-SUPERVISIONADO'])
            
            with tab_supervised:
                st.subheader('Aprendizado supervisionado prevê com dados rotulados pré-definidos')
                
                tab_classification, tab_regression = st.tabs(['CLASSIFICAÇÃO', 'REGRESSÃO'])
                
                with tab_classification:
                    st.markdown('Classificação (Classifier) - RandomForestClassifier')
                    st.caption('Input')
                    st.code("""X = [[0, 0], [1, 1], [0, 1], [1, 0]]
                    y = [0, 1, 1, 0]   # rótulos""", language='python')
                    st.caption('Processing')
                    st.code("""classificador = RandomForestClassifier(random_state=0)
                    classificador.fit(X, y)

                    y_pred = classificador.predict([[0, 0], [1, 1], [0, 1]])""", language='python')
                    st.caption('Output')
                    st.code('Predições: [0 1 1]', language='python')
                
                with tab_regression:
                    st.markdown('Regressão (Regressor) - RandomForestRegressor')
                    st.caption('Input')
                    st.code("""X = [[0, 0], [1, 1], [2, 2], [3, 3]]
                    y = [0.0, 1.0, 2.0, 3.0]   # valores contínuos""", language='python')
                    st.caption('Processing')
                    st.code("""regressor = RandomForestRegressor(random_state=0)
                    regressor.fit(X, y)

                    y_pred = regressor.predict([[1.5, 1.5], [2.5, 2.5]])""", language='python')
                    st.caption('Output')
                    st.code('Predições: [1.3 2.5]', language='python')
            
            with tab_unsupervised:
                st.subheader('Aprendizado não-supervisionado')
                st.subheader('Aprendizado não-supervisionado prevê GRUPOS sem dados rotulados pré-definidos')
                
                tab_clustering = st.tabs(['CLUSTERING'])[0]
                
                with tab_clustering:
                    st.markdown('Agrupamento (Clustering) – DBSCAN')
                    st.caption('Input')
                    st.code("""X = [[1, 2], [2, 2], [2, 3],
                        [8, 7], [8, 8], [25, 80]]""", language='python')
                    st.caption('Processing')
                    st.code('clusters = DBSCAN(eps=3, min_samples=2).fit(X)', language='python')
                    st.caption('Output')
                    st.code("""Labels atribuídos: [ 0  0  0  1  1 -1 ]
                    # -1 significa ponto considerado ruído (outlier)""", language='python')

    with tab_estimators:
        st.subheader("Tabela `estimators.tsv`")
        st.markdown("Este arquivo contém o catálogo de todos os estimadores (modelos e transformadores) do Scikit-learn que o agente pode utilizar.")
        
        df = pd.read_csv(st.session_state.files.get('estimators'), sep='\t', engine='python')
        
        column_config = {'exemplo_text': st.column_config.TextColumn('Exemplo', width='large')}
        
        colunas_string_para_list = ['class_path', 'params_list', 'submethods_list', 'compatible_scores', 'output_y_types']
        for column in colunas_string_para_list:
            df[column] = convert_column_to_list(df[column])
            column_config[column] = st.column_config.ListColumn(column, width='medium')
        
        st.dataframe(df, column_config=column_config, width='stretch')

    with tab_params:
        st.subheader("Tabela `parameters.tsv`")
        st.markdown("Este arquivo detalha as regras de otimização para cada hiperparâmetro.")
        
        df = pd.read_csv(st.session_state.files.get('parameters'), sep='\t', engine='python')
        
        df = convert_inf_values(df)
        
        column_config = {}
        
        colunas_string_para_list = ['param_list', 'estimators_list', 'param_dtype']
        for column in colunas_string_para_list:
            df[column] = convert_column_to_list(df[column])
            column_config[column] = st.column_config.ListColumn(column, width='small')
        
        st.dataframe(df, column_config=column_config, width='stretch')

    with tab_cronograma:
        st.subheader("Cronograma Geral do Projeto")
        show_data_by_function(None, 'cronograma')

    with tab_registro:
        st.subheader("Registro Geral de Atividades")
        show_data_by_function(None, 'registro_de_atividades')