import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from io import StringIO
from pathlib import Path
import graphviz

# --- Funções auxiliares (sem alterações) ---

def map_dtype(dtype):
    if np.issubdtype(dtype, np.integer):
        return 'int'
    elif np.issubdtype(dtype, np.floating):
        return 'float'
    return 'object'

@st.cache_data
def load_methods():
    path = "sklearn_methods.tsv"
    if not os.path.exists(path):
        st.error(f"Arquivo de configuração de métodos ('{path}') não encontrado.")
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    numeric_cols = ['input_dim_min', 'input_dim_max', 'input_min', 'input_max', 'output_dim_min', 'output_dim_max']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['input_dim_min', 'input_dim_max'], inplace=True)
    return df

def find_target_column(df):
    target_keywords = ["target", "objetivo", "destino"]
    for col in df.columns:
        if any(keyword in col.lower() for keyword in target_keywords):
            return col
    return df.columns[-1]

def get_input_summary(df):
    if df is None or df.empty:
        return {'Linhas': 0, 'Colunas': 0, 'Tipos': 'N/A', 'Possui Nulos': False, 'Valor Mínimo': 'N/A', 'Valor Máximo': 'N/A'}
    n_rows, n_cols = df.shape
    types = ', '.join(sorted(list(set(map_dtype(dt) for dt in df.dtypes))))
    has_nulls = df.isnull().any().any()
    numeric_cols = df.select_dtypes(include=np.number)
    min_val = numeric_cols.min().min() if not numeric_cols.empty else 'N/A'
    max_val = numeric_cols.max().max() if not numeric_cols.empty else 'N/A'
    return {'Linhas': n_rows, 'Colunas': n_cols, 'Tipos': types, 'Possui Nulos': has_nulls, 'Valor Mínimo': min_val, 'Valor Máximo': max_val}

def filter_models_by_properties(models_df, summary, allowed_types):
    if summary['Colunas'] == 0 or not allowed_types:
        return pd.DataFrame()
    compatible_models = models_df.copy()
    compatible_models = compatible_models[compatible_models['input_tipo'].isin(allowed_types)]
    n_cols = summary['Colunas']
    compatible_models = compatible_models[(compatible_models['input_dim_min'] <= n_cols) & (compatible_models['input_dim_max'] >= n_cols)]
    if summary['Possui Nulos']:
        compatible_models = compatible_models[compatible_models['input_nulos'] == True]
    min_val, max_val = summary['Valor Mínimo'], summary['Valor Máximo']
    if isinstance(min_val, np.number) and isinstance(max_val, np.number):
        compatible_models = compatible_models.loc[(compatible_models['input_min'] <= min_val) & (compatible_models['input_max'] >= max_val)]
    return compatible_models

# --- Nova Lógica de Construção Visual do Pipeline ---

def generate_pipeline_graph(pipeline_state):
    dot = graphviz.Digraph(comment='Pipeline')
    dot.attr(rankdir='LR', splines='ortho', pagedir='BL')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightgrey')

    # 1. Nó de Input e Colunas
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input Dataset', style='filled', color='lightgrey')
        c.node('df_input', label=st.session_state.selected_dataset, shape='folder', fillcolor='white')

        # Reordenar colunas para visualização: target por último
        ordered_cols = [col for col in pipeline_state['all_columns'] if col != pipeline_state['target_col']]
        if pipeline_state['target_col']:
            ordered_cols.append(pipeline_state['target_col'])

        for col in ordered_cols:
            c.node(col, label=col, shape='ellipse', fillcolor='white')

    # 2. Módulos de Features e conexões
    feature_outputs = []
    if pipeline_state["feature_groups"]:
        with dot.subgraph(name='cluster_features') as c:
            c.attr(label='Pré-processamento / Features (N1)', style='filled', color='lightgrey')
            for i, group in enumerate(pipeline_state["feature_groups"]):
                module_node = f"module_{i}"
                output_node = f"output_{i}"
                feature_outputs.append(output_node)
                c.node(module_node, label=group["module"], fillcolor='lightblue')
                c.node(output_node, label=f"Saída {i+1}", shape='ellipse', fillcolor='white')
                dot.edge(module_node, output_node)
                for col in group["inputs"]:
                    dot.edge(col, module_node)

    # 3. Estimador Final e conexões
    if pipeline_state["final_estimator"]:
        with dot.subgraph(name='cluster_estimator') as c:
            c.attr(label='Estimador Final (N2)', style='filled', color='lightgrey')
            c.node('final_estimator', label=pipeline_state["final_estimator"], fillcolor='lightgreen')
            # Conectar outputs das features ao estimador
            for out in feature_outputs:
                dot.edge(out, 'final_estimator')
            # Conectar target ao estimador
            if pipeline_state["target_col"]:
                dot.edge(pipeline_state["target_col"], 'final_estimator', color='red', style='dashed')

            # 4. Nó de Output
            dot.node('output', 'Output', shape='folder', style='filled', fillcolor='gold')
            dot.edge('final_estimator', 'output')

    return dot

def plot_visual_pipeline_builder(df_input, methods_df):
    st.header("Construtor Visual de Pipeline")

    # --- Inicialização do Estado do Pipeline ---
    if 'pipeline' not in st.session_state or st.session_state.get("dataset_name") != st.session_state.selected_dataset:
        st.session_state.dataset_name = st.session_state.selected_dataset
        st.session_state.pipeline = {
            "all_columns": df_input.columns.tolist(),
            "target_col": find_target_column(df_input),
            "feature_groups": [],
            "final_estimator": None
        }
    pipeline_state = st.session_state.pipeline

    # --- Layout: Painel de Controle e Área de Visualização ---
    control_col, viz_col = st.columns([1, 2])

    with control_col:
        st.subheader("Painel de Controle")

        # --- 1. Configuração do Alvo (y) ---
        with st.expander("1. Definir Coluna Alvo (y)", expanded=True):
            pipeline_state["target_col"] = st.selectbox(
                "Coluna Alvo (y)",
                options=pipeline_state['all_columns'],
                index=pipeline_state['all_columns'].index(pipeline_state["target_col"])
            )

        # --- 2. Módulos de Pré-processamento (N1) ---
        with st.expander("2. Adicionar Módulos de Features (N1)", expanded=True):
            st.write("Crie grupos de colunas para conectar a um módulo de processamento.")
            for i, group in enumerate(pipeline_state["feature_groups"]):
                st.markdown(f"**Grupo {i+1}**")
                group["inputs"] = st.multiselect(f"Entradas para Grupo {i+1}", options=[c for c in pipeline_state['all_columns'] if c != pipeline_state["target_col"]], default=group["inputs"], key=f"ms_{i}")
                if group["inputs"]:
                    group_df = df_input[group["inputs"]]
                    summary = get_input_summary(group_df)
                    types = [map_dtype(dt) for dt in group_df.dtypes.unique()]
                    compatible_modules = filter_models_by_properties(methods_df, summary, types)
                    group["module"] = st.selectbox(f"Módulo para Grupo {i+1}", options=compatible_modules['nome'].tolist(), key=f"sb_{i}")
            if st.button("Adicionar Grupo de Features"):
                st.session_state.pipeline["feature_groups"].append({"inputs": [], "module": None})
                st.rerun()

        # --- 3. Estimador Final (N2) ---
        with st.expander("3. Definir Estimador Final (N2)", expanded=True):
            # Filtra apenas modelos de Classificação/Regressão para o final
            final_models = methods_df[methods_df['processamento'].isin(["Classificação", "Regressão"])]
            pipeline_state["final_estimator"] = st.selectbox(
                "Estimador Final",
                options=[""] + final_models['nome'].tolist(),
                key="final_estimator"
            )

    with viz_col:
        st.subheader("Visualização do Pipeline")
        st.graphviz_chart(generate_pipeline_graph(pipeline_state))

# --- Função principal da página de Datasets ---

DATA_DIR = Path("data")
DATASETS_TSV = DATA_DIR / "datasets_metadata.tsv"

def datasets_scikit_learn():
    st.header("Seleção de Dataset e Construção de Pipeline")

    df_meta = pd.read_csv(DATASETS_TSV, sep="\t", index_col=0)
    df_meta["Baixado"] = df_meta.index.to_series().apply(lambda x: (DATA_DIR / f"{x}.csv").exists())

    if "selected_dataset" not in st.session_state:
        st.session_state["selected_dataset"] = None

    st.info("Selecione uma linha na tabela abaixo para carregar um dataset e iniciar o construtor de pipelines.")
    event = st.dataframe(
        df_meta,
        on_select="rerun",
        selection_mode="single-row",
    )

    if event.selection.rows:
        st.session_state.selected_dataset = df_meta.index[event.selection.rows[0]]
    
    if st.session_state.selected_dataset:
        st.success(f"Dataset Ativo: **{st.session_state.selected_dataset}**")
        if st.button("Limpar Seleção"):
            st.session_state.selected_dataset = None
            st.rerun()

        file_path = DATA_DIR / f"{st.session_state.selected_dataset}.csv"
        if not file_path.exists():
            st.error(f"Arquivo {file_path} não encontrado. A lógica de download precisa ser implementada.")
            return

        df_input = pd.read_csv(file_path)
        plot_visual_pipeline_builder(df_input, load_methods())