import streamlit as st
import pandas as pd
import numpy as np
import graphviz as gv
#from sklearn import datasets


def df_select_single_row(df):
    """Retorna um DataFrame interativo para seleção de linhas."""
    event = st.dataframe(
        df,
        width="stretch",
        height=300,
        hide_index=False,
        on_select="rerun",
        selection_mode='single-row',
    )
    
    selected_event = event.get('selection', None)
    if selected_event is not None:
        rows = selected_event.get('rows', None)
        if rows is not None and len(rows) > 0 and rows[0] is not None:
            dataset_name = df.index[rows[0]]
            return dataset_name
    st.info("Clique na primeira coluna para selecionar uma linha")
    return None


def build_feature_table(X: pd.DataFrame):
    """Cria tabela de resumo das features, com roles e estatísticas básicas."""

    desc = X.describe(include="all").transpose()

    # Contagem de nulos e não-nulos
    nulls = X.isnull().sum()
    non_nulls = X.notnull().sum()

    # Mediana
    medians = X.median(numeric_only=True)

    # Tipo da coluna
    dtypes = X.dtypes.astype(str)

    # Normalização para o gráfico
    normalized = (X - X.min()) / (X.max() - X.min())

    # Monta dataframe final
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
        use_container_width=True,
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
        use_container_width=True,
        num_rows="dynamic",
    )

    return edited
