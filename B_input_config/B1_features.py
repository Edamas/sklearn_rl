import streamlit as st
import pandas as pd
from functions import build_feature_table
from functions import log_message

def feature_definition():
    df = st.session_state.get("original_df")
    if df is None:
        return

    st.subheader("2. Definição de Features e Alvo (Target)")

    stats = build_feature_table(df)
    edited_stats = st.data_editor(
        stats,
        column_config={
            "Feature Role": st.column_config.SelectboxColumn("Feature Role", options=["X", "y", "desativado"]),
            "Gráfico": st.column_config.LineChartColumn("Gráfico"),
        },
        hide_index=False,
        num_rows="fixed",
    )

    X_cols = edited_stats[edited_stats["Feature Role"] == "X"]["Coluna"].tolist()
    y_cols = edited_stats[edited_stats["Feature Role"] == "y"]["Coluna"].tolist()

    if not X_cols:
        log_message("WARNING", "Selecione pelo menos uma coluna como 'X' (feature) para continuar.")
        st.stop()
    
    st.session_state.X_cols = X_cols
    st.session_state.y_cols = y_cols