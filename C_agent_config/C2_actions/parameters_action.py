import streamlit as st
import pandas as pd
from .utils import convert_column_to_list, convert_inf_values

def show_parameters():
    st.subheader("Parâmetros dos Estimadores do Agente:")
    st.markdown("""
        **Parâmetros** são configurações que controlam o comportamento dos estimadores:
        - **Hiperparâmetros**: Controlam o processo de aprendizado
        - **Limites**: Definem valores mínimos e máximos para otimização
        - **Opções**: Listas de valores possíveis para teste e validação
        
        A configuração adequada dos parâmetros é crucial para o desempenho dos modelos.
        """)
    metadata_tab, view_tab = st.tabs(['Metadados', 'Exibição'])
    with metadata_tab:
        st.markdown('`parameters.tsv`')
        st.markdown('''
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
        | `from_sklearn_docs` | Booleano | Indica se o registro foi preenchido automaticamente a partir da documentação do Scikit-learn. | `True` |''')

    with view_tab:
        st.markdown('`parameters.tsv`')
        df = pd.read_csv(st.session_state.files.get('parameters'), sep='\t')
        
        # Converte valores infinitos usando a mesma função
        df = convert_inf_values(df)
        
        # Inicialização da configuração das colunas
        column_config = {}
        
        # conversão de colunas para list - estrutura manual como solicitado
        colunas_string_para_list = ['param_list', 'estimators_list', 'param_dtype']  # Adicione outras colunas se necessário
        for column in colunas_string_para_list:
            df[column] = convert_column_to_list(df[column])
            column_config[column] = st.column_config.ListColumn(column, width='small')
        
        st.dataframe(
            df,
            column_config=column_config,
            width='stretch'
        )

