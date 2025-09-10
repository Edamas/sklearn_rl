# sklearn_methods_app.py
import streamlit as st
import pandas as pd
import ast

METHODS_TSV = st.session_state.files.get('sklearn_methods')
CATEGORIES_TSV = st.session_state.files.get('categorias_sklearn')
ESTIMATORS_TSV = st.session_state.files.get('estimators')
PARAMETERS_TSV = st.session_state.files.get('parameters')

# -----------------------------
# Função para exibir tabela de métodos
# -----------------------------
def show_sklearn_methods():
    """
    Exibe a tabela de métodos do Scikit-learn com gradiente de cores para colunas numéricas.
    """
    try:
        df = pd.read_csv(METHODS_TSV, sep="\t", index_col=0, decimal=",")
    except Exception as e:
        st.error(f"Erro ao ler o TSV: {e}")
        return

    st.subheader("Tabela de Métodos do Scikit-Learn")

    # Detecta todas as colunas numéricas
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # Preenche valores nulos por 0.0
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    # Exibe com gradiente de cores (viridis) e apenas 1 casa decimal
    st.dataframe(
        df.style.background_gradient(cmap="viridis", subset=numeric_cols).format(precision=1),
        width="stretch")   # <--- substitui use_container_width=True

# -----------------------------
# Função para exibir categorias
# -----------------------------
def show_sklearn_categories():
    """
    Exibe categorias de métodos do Scikit-learn em duas colunas usando TSV.
    """
    try:
        df = pd.read_csv(CATEGORIES_TSV, sep="\t")
    except Exception as e:
        st.error(f"Erro ao ler o TSV de categorias: {e}")
        return

    st.subheader("Categorias de Métodos do Scikit-Learn")
    st.markdown("---")
    for _, row in df.iterrows():
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.subheader(f"{row['Categoria']}")
            
        with col2:
            st.markdown(f"Descrição Sintetizada")
            st.markdown(f"`{row.get('Descricao_Sintetica','')}`")
        with col3:
            st.markdown(f"Descrição Completa / Principais Métodos:")
            st.markdown(f"`{row.get('Descricao_Completa','')}`")
        with col4:
            st.markdown(f"Exemplo Prático Simplificado:")
            st.markdown(f"`{row.get('Exemplo','')}`")
        st.markdown("---")

# -----------------------------
# Função para exibir estimadores
# -----------------------------
def show_estimators():
    st.subheader("Tabela de Estimadores do Agente")
    try:
        df = pd.read_csv(ESTIMATORS_TSV, sep='\t')
        
        # Converte as colunas de string para listas reais
        df['params_list'] = df['params_list'].apply(
            lambda x: [item.strip().strip("'\"") for item in x.strip('[]').split(',')] if isinstance(x, str) and x.startswith('[') else []
        )
        df['submethods_list'] = df['submethods_list'].apply(
            lambda x: [i.strip() for i in x.split(',')] if isinstance(x, str) else []
        )

        st.dataframe(
            df,
            column_config={
                "params_list": st.column_config.ListColumn("Parâmetros", width="medium"),
                "submethods_list": st.column_config.ListColumn("Submétodos", width="large"),
            },
            width='stretch'
        )
    except FileNotFoundError:
        st.error(f"Arquivo {ESTIMATORS_TSV} não encontrado.")
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")

# -----------------------------
# Função para exibir parâmetros
# -----------------------------
def show_parameters():
    st.subheader("Tabela de Parâmetros do Agente")
    try:
        df = pd.read_csv(PARAMETERS_TSV, sep='\t')
        
        # Converte a coluna de string para lista real
        df['param_list'] = df['param_list'].apply(
            lambda x: [item.strip().strip("'\"") for item in x.strip('[]').split(',')] if isinstance(x, str) and x.startswith('[') else []
        )

        st.dataframe(
            df,
            column_config={
                "param_list": st.column_config.ListColumn("Valores Possíveis", width="medium"),
            },
            width='stretch'
        )
    except FileNotFoundError:
        st.error(f"Arquivo {PARAMETERS_TSV} não encontrado.")
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")