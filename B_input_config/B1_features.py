import streamlit as st
import pandas as pd
from functions import log_message

def feature_definition():
    df = st.session_state.get("original_df")
    if df is None:
        return

    st.subheader("2. Definição da(s) Coluna(s) Alvo (Target)")

    # Auto-detect potential target columns
    potential_targets = ['target', 'Target', 'y', 'Y', 'destino', 'Destino', 'class', 'Class', 'classe', 'Classe']
    detected_targets = [col for col in df.columns if col in potential_targets]

    # Allow user to select one or more target columns (y)
    y_cols = st.multiselect(
        "Selecione a(s) coluna(s) que você quer prever (alvo)",
        options=df.columns.tolist(),
        default=detected_targets,
        help="Selecione uma ou mais colunas para um aprendizado supervisionado. Deixe em branco para um aprendizado não supervisionado (ex: clusterização)."
    )

    # Define X and y columns
    if y_cols:
        X_cols = [col for col in df.columns if col not in y_cols]
        
        # Determine task type for supervised learning
        # Simple heuristic: if all targets are numeric and have many unique values, it's regression. Otherwise, classification.
        is_regression = True
        for col in y_cols:
            if not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() < 20:
                is_regression = False
                break
        
        task_type = "Regression" if is_regression else "Classification"
        st.metric("Tipo de Tarefa Inferido", task_type)

    else: # Unsupervised
        X_cols = df.columns.tolist()
        task_type = "Unsupervised"
        st.info("Nenhuma coluna alvo selecionada. O agente executará tarefas não supervisionadas (ex: Clusterização, Detecção de Anomalias).")

    # Store in session state
    st.session_state.X_cols = X_cols
    st.session_state.y_cols = y_cols
    st.session_state.task_type = task_type
