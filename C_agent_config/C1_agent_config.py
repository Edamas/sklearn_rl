import streamlit as st
import pandas as pd
from pathlib import Path
from functions import log_message, df_select_rows

ESTIMATORS_FILE = st.session_state.files.get('estimators')

def agent_configuration():
    if not ESTIMATORS_FILE:
        st.error("O caminho para 'estimators.tsv' não foi encontrado. Verifique a chave 'estimators' em 'files.tsv'.")
        st.stop()

    X_cols = st.session_state.get("X_cols")
    if X_cols is None:
        return

    st.subheader("3. Configuração do Agente")
    y_cols = st.session_state.get("y_cols")
    task_type = st.session_state.get("task_type")

    try:
        estimators_df = pd.read_csv(ESTIMATORS_FILE, sep='\t')
    except FileNotFoundError as e:
        log_message("EXCEPTION", f"Arquivo de estimadores não encontrado em '{ESTIMATORS_FILE}'.", exception=e)
        st.stop()

    estimators_df['input_y_structure'] = estimators_df['input_y_structure'].fillna('None')
    estimators_df['input_y_types'] = estimators_df['input_y_types'].fillna('None')
    estimators_df['apt_for_training'] = estimators_df['apt_for_training'].astype(bool)

    n_features = len(X_cols)
    n_targets = len(y_cols)

    dataset_X_structure = "(n_samples, n_features)"
    dataset_X_types_list = ["float", "int"]

    if task_type in ["Classification", "Regression"]:
        dataset_y_structure = "(n_samples,)" if n_targets == 1 else "(n_samples, n_outputs)"
        dataset_y_types_list = ["float", "int"]
    else: # Unsupervised
        dataset_y_structure = "None"
        dataset_y_types_list = ["None"]

    def check_types(cell_value, valid_types):
        # Remove brackets and split by comma, then strip whitespace
        parsed_types = [t.strip() for t in str(cell_value).replace('[', '').replace(']', '').split(',')]
        return any(req_type in valid_types for req_type in parsed_types)

    compatible_estimators = estimators_df[
        (estimators_df['X_min'] <= n_features) &
        (estimators_df['X_max'] >= n_features) &
        (estimators_df['y_min'] <= n_targets) &
        (estimators_df['y_max'] >= n_targets) &
        (estimators_df['apt_for_training'] == True) &
        (estimators_df['input_X_structure'].str.contains(dataset_X_structure, na=False, regex=False)) &
        (estimators_df['input_X_types'].apply(lambda x: check_types(x, dataset_X_types_list))) &
        (estimators_df['input_y_structure'].str.contains(dataset_y_structure, na=False, regex=False)) &
        (estimators_df['input_y_types'].apply(lambda x: check_types(x, dataset_y_types_list)))
    ]

    if task_type == "Classification":
        compatible_estimators = compatible_estimators[compatible_estimators['estimator_type'] == 'Classifier']
    elif task_type == "Regression":
        compatible_estimators = compatible_estimators[compatible_estimators['estimator_type'] == 'Regressor']
    elif task_type == "Unsupervised":
        compatible_estimators = compatible_estimators[
            (compatible_estimators['estimator_type'] == 'Transformer') |
            (compatible_estimators['estimator_type'] == 'Cluster')
        ]

    st.write(f"Foram encontrados **{len(compatible_estimators)}** estimadores compatíveis com a sua configuração de dados (Features: {n_features}, Alvos: {n_targets}, Tarefa: {task_type}).")

    if compatible_estimators.empty:
        log_message("ERROR", "Nenhum estimador compatível encontrado. Ajuste a seleção de features/alvos ou o arquivo de estimadores.")
        st.stop()

    st.markdown("### 3.1 Selecione o Estimador para Avaliação")
    st.info("Selecione um ou mais estimadores compatíveis na tabela abaixo.")
    
    selected_indices = df_select_rows(
        compatible_estimators[['estimator_name', 'estimator_type', 'category']],
        prompt="Selecione um ou mais estimadores na tabela acima."
    )

    if not selected_indices:
        log_message("WARNING", "Nenhum estimador selecionado para avaliação. Por favor, selecione pelo menos um.", display_streamlit=True)
        st.stop()

    selected_estimators_df = compatible_estimators.loc[selected_indices]
    selected_estimator_names = selected_estimators_df["estimator_name"].tolist()

    st.session_state.compatible_estimators = selected_estimators_df
    st.session_state.selected_estimator_names = selected_estimator_names

    num_episodes = st.slider(
        "Episódios de Treinamento",
        min_value=1,
        max_value=1000,
        value=min(10, 1000),
        help="Número total de pipelines que o agente criará e avaliará.",
        key="num_episodes_slider"
    )
    st.session_state.num_episodes = num_episodes
