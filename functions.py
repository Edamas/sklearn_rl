import streamlit as st
import pandas as pd
import numpy as np
import graphviz as gv
from typing import Optional, Literal

def df_select_rows(df, selection_mode: Literal['single-row', 'multi-row'] = 'multi-row', prompt: Optional[str] = None, key: str = "dataframe_selection"):
    """
    NÃO MODIFICAR ESSA FUNÇÃO, POIS ELA É IMPORTANTE PARA TODOS OS DATAFRAMES SELECIONÁVEIS DO PROJETO.
    Retorna um DataFrame interativo para seleção de linhas.
    O modo de seleção pode ser 'single-row' ou 'multi-row'.
    A seleção é armazenada em st.session_state[key].
    """
    event = st.dataframe(
        df,
        width="stretch",
        height=300,
        hide_index=False,
        on_select="rerun",
        selection_mode=selection_mode,
        key=key # Add a key to the dataframe
    )
    
    # Get selection from session state
    selection = event.get('selection')

    if selection and selection.get('rows'):
        row_indices = selection['rows']
        if not row_indices:
            return None if selection_mode == 'single-row' else []

        if selection_mode == 'single-row':
            return df.index[row_indices[0]]
        else: # multi-row
            return df.index[row_indices].tolist()

    if prompt:
        st.info(prompt)
    
    return None if selection_mode == 'single-row' else []

def analyze_and_group_columns(df: pd.DataFrame):
    """
    Analyzes a DataFrame, groups columns by data type, and calculates summary statistics for each group.

    Returns:
        A DataFrame summarizing the column groups.
    """
    # Attempt to convert object columns to datetime
    df_temp = df.copy()
    for col in df_temp.select_dtypes(include=['object']).columns:
        try:
            df_temp[col] = pd.to_datetime(df_temp[col], errors='coerce')
        except (ValueError, TypeError):
            continue # Ignore columns that can't be converted

    groups = {
        'Numeric': [],
        'Binary': [],
        'Categorical': [],
        'Date': [],
        'Text': [] # For high cardinality strings
    }

    for col in df_temp.columns:
        dtype = df_temp[col].dtype
        nunique = df_temp[col].nunique()

        if pd.api.types.is_numeric_dtype(dtype):
            if nunique == 2:
                groups['Binary'].append(col)
            else:
                groups['Numeric'].append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            groups['Date'].append(col)
        elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            # Heuristic for categorical vs. text
            if nunique < 50 and nunique > 0:
                groups['Categorical'].append(col)
            else: 
                groups['Text'].append(col)

    summary_list = []
    for group_name, cols in groups.items():
        if not cols:
            continue

        n_cols = len(cols)
        subset = df_temp[cols]
        n_rows = len(subset)
        nulls = subset.isnull().sum().sum()
        pct_nulls = nulls / (n_rows * n_cols) if (n_rows * n_cols) > 0 else 0

        stats = {
            'group_type': group_name,
            'column_count': n_cols,
            'columns': cols,
            'null_percentage': pct_nulls
        }

        if group_name == 'Numeric':
            stats['mean_of_means'] = subset.mean().mean()
            stats['mean_of_stds'] = subset.std().mean()
            stats['min_of_mins'] = subset.min().min()
            stats['max_of_maxs'] = subset.max().max()
        elif group_name == 'Binary':
            if n_cols > 0:
                counts = subset[cols[0]].value_counts(normalize=True)
                stats['value1_proportion'] = counts.iloc[0] if len(counts) > 0 else 0.0
                stats['value2_proportion'] = counts.iloc[1] if len(counts) > 1 else 0.0
        elif group_name == 'Categorical':
            stats['mean_unique_values'] = subset.nunique().mean()
        elif group_name == 'Text':
            stats['mean_unique_values'] = subset.nunique().mean()
            stats['mean_string_length'] = subset.astype(str).apply(lambda x: x.str.len()).mean().mean()
        elif group_name == 'Date':
            min_date = subset.min().min() if not subset.min().empty else pd.NaT
            max_date = subset.max().max() if not subset.max().empty else pd.NaT
            stats['min_date'] = min_date
            stats['max_date'] = max_date

        summary_list.append(stats)

    if not summary_list:
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary_list).set_index('group_type')
    return summary_df.fillna(0)

def build_feature_table(X: pd.DataFrame):
    """Cria tabela de resumo das features, com roles e estatísticas básicas."""
    desc = X.describe(include="all").transpose()
    nulls = X.isnull().sum()
    non_nulls = X.notnull().sum()
    medians = X.median(numeric_only=True)
    dtypes = X.dtypes.astype(str)
    normalized = (X - X.min()) / (X.max() - X.min())
    summary = pd.DataFrame({
        "Feature Role": ["X"] * len(X.columns),
        "Coluna": X.columns,
        "Mínimo": desc["min"].fillna(""),
        "Média": desc["mean"].fillna(""),
        "Mediana": medians.reindex(X.columns).fillna(""),
        "Máximo": desc["max"].fillna(""),
        "Desvio Padrão": desc["std"].fillna(""),
        "Nulos": nulls,
        "Não-Nulos": non_nulls,
        "Tipo": dtypes,
        "Gráfico": [normalized[col].tolist() for col in X.columns],
    })
    return summary

def draw_graph(feature_table: pd.DataFrame, has_target: bool):
    """Desenha grafo com Graphviz baseado nos papéis das features."""
    dot = gv.Digraph()
    dot.attr(rankdir="LR")
    for col, role in zip(feature_table["Coluna"], feature_table["Feature Role"]):
        if role == "X":
            dot.node(col, shape="box", color="lightblue")
            dot.edge(col, "Agente")
        elif role == "y":
            dot.node(col, shape="ellipse", color="lightgreen")
            dot.edge(col, "Rótulo")
    dot.node("Agente", shape="box", style="filled", color="lightgray")
    dot.node("Rótulo", shape="oval", style="filled", color="lightyellow")
    return dot

def show_feature_table(X: pd.DataFrame):
    """Mostra tabela interativa com Feature Role, estatísticas + gráfico normalizado."""
    st.subheader("Resumo do Dataset Selecionado")
    col1, col2 = st.columns([3, 2])
    with col2:
        st.info("Selecione o papel (Feature role) de cada coluna do dataset escolhido")
    feature_table = build_feature_table(X)
    edited = st.data_editor(
        feature_table,
        column_config={
            "Feature Role": st.column_config.SelectboxColumn(
                "Feature Role", options=["X", "y", "[Desativado]"], default="X"
            ),
            "gráfico": st.column_config.LineChartColumn("Distribuição Normalizada"),
        },
        hide_index=True,
        width='stretch',
        disabled=[
            "Coluna", "min", "mean", "median",
            "max", "std", "nulos", "nao_nulos", "dtype"
        ],
    )
    return edited

def show_feature_editor(X: pd.DataFrame):
    """Exibe editor de features com selectbox, estatísticas e gráfico normalizado."""
    st.subheader("📊 Resumo do dataset selecionado")
    st.write("➡️ Selecione o papel (**Feature role**) de cada coluna do dataset escolhido")
    edited = st.data_editor(
        build_feature_table(X),
        column_config={
            "Feature Role": st.column_config.SelectboxColumn(
                "Feature Role",
                help="Defina se a coluna é entrada (X), alvo (y) ou deve ser ignorada.",
                options=["X", "y", "[Desativado]"],
                default="X",
            ),
            "Coluna": st.column_config.TextColumn("Coluna", disabled=True),
            "Mínimo": st.column_config.NumberColumn("Mínimo", disabled=True),
            "Média": st.column_config.NumberColumn("Média", disabled=True),
            "Mediana": st.column_config.NumberColumn("Mediana", disabled=True),
            "Máximo": st.column_config.NumberColumn("Máximo", disabled=True),
            "Desvio Padrão": st.column_config.NumberColumn("Desvio Padrão", disabled=True),
            "Nulos": st.column_config.NumberColumn("Nulos", disabled=True),
            "Não-Nulos": st.column_config.NumberColumn("Não-Nulos", disabled=True),
            "Tipo": st.column_config.TextColumn("Tipo", disabled=True),
            "Gráfico": st.column_config.LineChartColumn("Gráfico", width="medium"),
        },
        hide_index=True,
        width='stretch',
        num_rows="dynamic",
    )
    return edited


import os
from pathlib import Path

def create_optimized_estimators_tsv(original_tsv_path: str, output_tsv_path: str):
    """
    Reads the original estimators.tsv, selects only the actively used columns,
    and saves them to a new optimized TSV file.
    """
    st.toast(f"Criando arquivo otimizado de estimadores: {output_tsv_path}")

    actively_used_columns = [
        'estimator_name', 'class_path', 'estimator_type', 'input_X_structure',
        'input_X_types', 'input_y_structure', 'input_y_types', 'output_X_structure',
        'output_X_types', 'output_y_structure', 'output_y_types', 'compatible_scores'
    ]

    try:
        # Read the original TSV file
        df_original = pd.read_csv(original_tsv_path, sep='\t', low_memory=False)

        # Select only the actively used columns
        df_optimized = df_original[actively_used_columns]

        # Ensure the output directory exists
        output_dir = Path(output_tsv_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the optimized DataFrame to a new TSV file
        df_optimized.to_csv(output_tsv_path, sep='\t', index=False)
        st.toast(f"Arquivo otimizado de estimadores criado com sucesso em: {output_tsv_path}")
        return True
    except FileNotFoundError:
        st.error(f"Erro: O arquivo original de estimadores '{original_tsv_path}' não foi encontrado.")
        return False
    except KeyError as e:
        st.error(f"Erro: Coluna essencial faltando no arquivo original de estimadores: {e}. Verifique se '{original_tsv_path}' está completo.")
        return False
    except Exception as e:
        st.error(f"Ocorreu um erro ao criar o arquivo otimizado de estimadores: {e}")
        return False


