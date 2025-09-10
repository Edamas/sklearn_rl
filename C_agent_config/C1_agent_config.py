import streamlit as st
import pandas as pd
from pathlib import Path
from functions import log_message, df_select_rows

ESTIMATORS_FILE = st.session_state.files.get('estimators')

def agent_configuration():
    X_cols = st.session_state.get("X_cols")

    # Guard: Only show this section if features have been defined
    if X_cols is None:
        return

    st.subheader("3. Configuração do Agente")
    y_cols = st.session_state.get("y_cols")

    try:
        estimators_df = pd.read_csv(ESTIMATORS_FILE, sep='\t')
    except FileNotFoundError as e:
        log_message("EXCEPTION", f"Arquivo de estimadores não encontrado em '{ESTIMATORS_FILE}'.", exception=e)
        st.stop()

    # FIX: Trata valores nulos (NaN) como a string 'None' para compatibilidade
    estimators_df['input_y_structure'] = estimators_df['input_y_structure'].fillna('None')
    estimators_df['input_y_types'] = estimators_df['input_y_types'].fillna('None')

    n_features = len(X_cols)
    n_targets = len(y_cols)

    # Garante que a coluna booleana seja do tipo bool
    estimators_df['apt_for_training'] = estimators_df['apt_for_training'].astype(bool)

    # Estruturas padrão dos datasets (simples, só numérico por enquanto)
    dataset_X_structure = "(n_samples, n_features)"
    dataset_X_types_list = ["float", "int"]  # Assumindo dados numéricos

    if y_cols:
        if len(y_cols) == 1:
            dataset_y_structure = "(n_samples,)"
        else:
            dataset_y_structure = "(n_samples, n_outputs)"
        dataset_y_types_list = ["float", "int"]
    else:
        dataset_y_structure = "None"
        dataset_y_types_list = ["None"]

    # Funções seguras de verificação
    def check_types(cell_value, valid_types):
        """Converte valor em string, divide e verifica compatibilidade"""
        return any(req_type in valid_types for req_type in str(cell_value).split(','))

    compatible_estimators = estimators_df[
        (estimators_df['X_min'] <= n_features) &
        (estimators_df['X_max'] >= n_features) &
        (estimators_df['y_min'] <= n_targets) &
        (estimators_df['y_max'] >= n_targets) &
        (estimators_df['apt_for_training'] == True) &
        (estimators_df['input_X_structure'] == dataset_X_structure) &
        (estimators_df['input_X_types'].apply(lambda x: check_types(x, dataset_X_types_list))) &
        (estimators_df['input_y_structure'] == dataset_y_structure) &
        (estimators_df['input_y_types'].apply(lambda x: check_types(x, dataset_y_types_list)))
    ]

    # Add filter for estimator_type based on whether y_cols is present
    if y_cols: # Supervised task, expect Classifier or Regressor
        compatible_estimators = compatible_estimators[
            (compatible_estimators['estimator_type'] == 'Classifier') |
            (compatible_estimators['estimator_type'] == 'Regressor')
        ]
    else: # Unsupervised task, expect Transformer or Cluster
        compatible_estimators = compatible_estimators[
            (compatible_estimators['estimator_type'] == 'Transformer') |
            (compatible_estimators['estimator_type'] == 'Cluster')
        ]

    st.write(f"Foram encontrados **{len(compatible_estimators)}** estimadores compatíveis com a sua configuração de dados (Features: {n_features}, Alvos: {n_targets}).")

    if compatible_estimators.empty:
        log_message("ERROR", "Nenhum estimador compatível encontrado. Ajuste a seleção de features/alvos ou o arquivo de estimadores.")
        st.stop()

    st.markdown("#### Selecione o Estimador para Avaliação")
    st.info("Selecione um estimador compatível na tabela abaixo.")
    # Use the imported function to get a single selection
    # Use the imported function to get multiple selections
    selected_indices = df_select_rows(
        compatible_estimators[['estimator_name', 'estimator_type', 'category']],
        prompt="Selecione um ou mais estimadores na tabela acima."
    )

    if not selected_indices:
        log_message("WARNING", "Nenhum estimador selecionado para avaliação. Por favor, selecione pelo menos um.", display_streamlit=True)
        st.stop()

    # Create a DataFrame with only the selected estimators
    selected_estimators_df = compatible_estimators.loc[selected_indices]
    selected_estimator_names = selected_estimators_df["estimator_name"].tolist()

    # Store the filtered DF and names
    st.session_state.compatible_estimators = selected_estimators_df
    st.session_state.selected_estimator_names = selected_estimator_names

    num_episodes = st.slider(
        "Episódios",
        min_value=1,
        max_value=1000, # Increased max value for total episodes
        value=min(10, 1000), # Default to 10 episodes
        help="Número total de pipelines que o agente criará e aplicará ao dataset (episódios).",
        key="num_episodes_slider"
    )
    st.session_state.num_episodes = num_episodes
