import streamlit as st
import pandas as pd
from functions import df_select_rows
from D_training.data_loader import load_estimators

def agent_configuration():
    X_cols = st.session_state.get("X_cols")
    if X_cols is None:
        return

    st.subheader("2. Processing")
    st.markdown("### Configuração do Agente")
    y_cols = st.session_state.get("y_cols")
    task_type = st.session_state.get("task_type")

    estimators_df = load_estimators()

    n_features = len(X_cols)
    n_targets = len(y_cols) if y_cols is not None else 0

    # Determine dataset types from column summary
    column_summary = st.session_state.get("column_summary_df")
    dataset_X_types = set()
    if column_summary is not None:
        if 'Numeric' in column_summary.index or 'Binary' in column_summary.index:
            dataset_X_types.update(['float', 'int'])
        if 'Categorical' in column_summary.index or 'Text' in column_summary.index:
            dataset_X_types.update(['str', 'object'])
    
    if not dataset_X_types:
        dataset_X_types = {'float', 'int'} # Default

    # Basic compatibility
    filter_basic = (
        (estimators_df['apt_for_training'] == True) &
        (estimators_df['X_min'] <= n_features) &
        (estimators_df['X_max'] >= n_features) &
        (estimators_df['y_min'] <= n_targets) &
        (estimators_df['y_max'] >= n_targets)
    )
    compatible_estimators_step1 = estimators_df[filter_basic].copy()

    # Type compatibility for X
    def check_input_types(estimator_types):
            # Ensure dataset_X_types are always lowercase strings
            dataset_X_types_lower = {t.lower() for t in dataset_X_types}

            if isinstance(estimator_types, list):
                # Convert list elements to lowercase strings for consistent comparison
                estimator_types_lower = {str(t).lower() for t in estimator_types}
                return bool(dataset_X_types_lower.intersection(estimator_types_lower))
            elif isinstance(estimator_types, str):
                # Handle single string type
                if estimator_types.lower() == 'any':
                    return True
                return estimator_types.lower() in dataset_X_types_lower
            return False

    compatible_estimators_step2 = compatible_estimators_step1[
        compatible_estimators_step1['input_X_types'].apply(check_input_types)
    ].copy() # Added .copy() to avoid SettingWithCopyWarning

    # New: Structure compatibility for X
    # Treat any tuple as compatible, and simplify other conditions
    compatible_estimators_step3 = compatible_estimators_step2.copy() # Start with a copy of step2 results

    final_structure_filter = compatible_estimators_step3['input_X_structure'].apply(
        lambda x: (isinstance(x, (tuple, list)) or # Any tuple or list is approved
                   (isinstance(x, str) and x.lower() == 'array-like') or
                   (isinstance(x, str) and ('n_features' in x.lower() or 'n_samples' in x.lower())) or
                   (x is None) or # If it's None, it's approved
                   (x == 0) # If it's 0, it's also approved for now, we'll filter later if needed
                  )
    )

    compatible_estimators_step3 = compatible_estimators_step3[final_structure_filter].copy()

    # Task-specific filtering
    if task_type == "Classification":
        compatible_estimators = compatible_estimators_step3[compatible_estimators_step3['estimator_type'] == 'Classifier']
    elif task_type == "Regression":
        compatible_estimators = compatible_estimators_step3[compatible_estimators_step3['estimator_type'] == 'Regressor']
    elif task_type == "Unsupervised":
        compatible_estimators = compatible_estimators_step3[
            compatible_estimators_step3['estimator_type'].isin(['Transformer', 'Cluster', 'CovarianceEstimator', 'OutlierDetector'])
        ]

    st.info(f"Foram encontrados **{len(compatible_estimators)}** estimadores compatíveis.")

    if compatible_estimators.empty:
        st.warning("Nenhum estimador compatível encontrado. Ajuste features/alvos ou arquivo de estimadores.")
        st.stop()

    st.markdown("### Selecione o Estimador para Avaliação")
    
    selected_indices = df_select_rows(
        compatible_estimators[['estimator_name', 'estimator_type', 'category']],
        prompt="Selecione estimadores compatíveis na tabela abaixo."
    )

    if not selected_indices:
        st.warning("Nenhum estimador selecionado. Selecione pelo menos um.")
        st.stop()

    selected_estimators_df = compatible_estimators.loc[selected_indices]
    selected_estimator_names = selected_estimators_df["estimator_name"].tolist()

    st.session_state.compatible_estimators = selected_estimators_df
    st.session_state.selected_estimator_names = selected_estimator_names

    num_episodes = st.slider(
        "Episódios de Treinamento",
        min_value=1,
        max_value=1000,
        value=100,
        help="Número total de pipelines que o agente criará e avaliará.",
        key="num_episodes_slider"
    )
    st.session_state.num_episodes = num_episodes