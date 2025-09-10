import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from D_training.agent_rl import AgentHyperparameterParser, create_estimator_instance
from pathlib import Path
from functions import log_message

import random

PARAMETERS_FILE = st.session_state.files.get('parameters')

def agent_training():
    # Guard: Only show this section if estimators have been selected
    if st.session_state.get("compatible_estimators") is None or st.session_state.get("num_episodes") is None:
        return

    st.subheader("4. Treinamento do Agente")

    # Get data from session state
    df = st.session_state.get("original_df")
    X_cols = st.session_state.get("X_cols")
    y_cols = st.session_state.get("y_cols") # Can be None
    compatible_estimators = st.session_state.get("compatible_estimators")
    selected_estimator_names = st.session_state.get("selected_estimator_names")
    num_episodes = st.session_state.get("num_episodes")
    dataset_name = st.session_state.get("dataset_name")
    # These are not set in the UI yet
    num_param_optimization_trials = st.session_state.get("num_param_optimization_trials", 10) # Default to 10
    num_models_to_evaluate = st.session_state.get("num_models_to_evaluate", 3) # Default to 3

    # Check for essential data. y_cols is not essential (unsupervised learning).
    if any(arg is None for arg in [df, X_cols, compatible_estimators, num_episodes, dataset_name]):
        log_message("WARNING", "Dados de configuração essenciais estão faltando. Por favor, revise as etapas anteriores.")
        st.stop()

    # Initialize session state for training status
    if 'training_started' not in st.session_state:
        st.session_state.training_started = False

    if st.button("🚀 Iniciar Treinamento", width='stretch'):
        if df is None:
            log_message("EXCEPTION", "O DataFrame 'df' está vazio ou não foi carregado corretamente.")
            st.stop()
        X = df[X_cols]
        y = df[y_cols] if y_cols else None

        try:
                        params_df = pd.read_csv(PARAMETERS_FILE, sep='\t')

        except FileNotFoundError as e:
            log_message("EXCEPTION", f"Arquivo de parâmetros '{PARAMETERS_FILE}' não encontrado.", exception=e)
            return

        # Trata o caso de y ser None (não supervisionado)
        if y is not None:
            y_raveled = y.values.ravel() # Ravel y here once
            X_train, _, y_train, _ = train_test_split(X, y_raveled, test_size=0.3, random_state=42)

            # Refined classification_task detection
            if pd.api.types.is_numeric_dtype(y_raveled):
                # Check if it's integer-like and has few unique values
                if pd.api.types.is_integer_dtype(y_raveled) and np.unique(y_raveled).size <= 50: # Increased threshold slightly
                    classification_task = True
                else:
                    classification_task = False # Numeric and not integer-like or too many unique values
            else:
                classification_task = True # Non-numeric (e.g., strings) are classification

            scoring_metric = "accuracy" if classification_task else "r2"
        else:
            X_train, _ = train_test_split(X, test_size=0.3, random_state=42)
            y_train = None
            classification_task = False
            scoring_metric = None # Usa o score padrão do estimador para clusterização/etc.

        # --- Inicia o processo de otimização ---
        parser = AgentHyperparameterParser(compatible_estimators, params_df)

        all_trials_results = []
        progress_area = st.container()
        overall_progress_bar = progress_area.progress(0, text="Progresso Geral da Avaliação de Modelos")

        tried_combinations = set() # To store (estimator_name, frozenset(params.items()))

        for i in range(num_episodes):
            # Randomly select an estimator for this trial
            estimator_def = compatible_estimators.sample(n=1).iloc[0]
            model_name = estimator_def['estimator_name']

            # Generate random parameters, ensuring uniqueness
            params = {}
            combination_key = None
            
            # Loop until a unique combination is found or max attempts reached
            max_attempts = 100 # Limit attempts to avoid infinite loops
            for attempt in range(max_attempts):
                params = parser.generate_random_params(estimator_def, estimator_name=model_name)
                combination_key = (model_name, frozenset(params.items()))
                if combination_key not in tried_combinations:
                    tried_combinations.add(combination_key)
                    break
                elif attempt == max_attempts - 1:
                    log_message("WARNING", f"Não foi possível gerar uma combinação única para {model_name} após {max_attempts} tentativas. Pulando este episódio.")
                    params = {} # Ensure empty params if no unique combination found
            
            if not params: # Skip if no unique params were generated
                continue

            start_time = time.time()
            model, error_msg = create_estimator_instance(estimator_def['class_path'], params)
            
            score = np.nan
            status = "Erro"
            pipeline_steps_repr = []
            if not error_msg:
                pipe = Pipeline([("model", model)])
                
                # Store pipeline representation with parameters
                for step_name, step_estimator in pipe.steps:
                    if hasattr(step_estimator, '__class__') and hasattr(step_estimator.__class__, '__name__'):
                        class_path = f"{step_estimator.__class__.__module__}.{step_estimator.__class__.__name__}"
                    else:
                        class_path = str(step_estimator) # Fallback for non-standard objects
                    pipeline_steps_repr.append((step_name, class_path, params)) # Include params here
                try:
                    estimator_type = estimator_def['estimator_type']
                    
                    if estimator_type in ['Classifier', 'Regressor']:
                        scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring=scoring_metric)
                    elif estimator_type in ['Transformer', 'Cluster']:
                        pipe.fit(X_train, y_train)
                        scores = [np.nan]
                    else:
                        log_message("WARNING", f"Tipo de estimador desconhecido: {estimator_type}. Apenas ajustando o modelo.", display_streamlit=False)
                        pipe.fit(X_train, y_train)
                        scores = [np.nan]

                    score = np.mean(scores)
                    status = "Sucesso"
                except Exception as e:
                    error_msg = str(e).replace('\n', ' ')
                    log_message("EXCEPTION", f"Erro durante cross_val_score para o modelo {model_name}.", exception=e, display_streamlit=False)

            end_time = time.time()
            duration = end_time - start_time
            
            trial_result = {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': duration,
                'estimator_name': model_name,
                'params': str(params), # Store params as string
                'status': status,
                'score': score if not np.isnan(score) else 0,
                'error': error_msg if error_msg else '',
                'pipeline_steps': pipeline_steps_repr # Add pipeline representation
            }
            all_trials_results.append(trial_result)

            progress_text = f"Episódio: {i+1}/{num_episodes} | Modelo: **{model_name}** | Score: {score:.3f} | Tempo: {duration:.2f}s"
            overall_progress_bar.progress((i + 1) / num_episodes, text=progress_text)

        
        if not all_trials_results:
            log_message("WARNING", "Nenhum resultado foi gerado pelo agente. Exibindo DataFrame vazio.")
            df_current_run = pd.DataFrame() # Create an empty DataFrame
        else:
            # Dataframe apenas com os resultados da execução atual
            df_current_run = pd.DataFrame(all_trials_results)

        # Salva os resultados no session_state para a página de resultados
        # Salva os resultados no session_state para a página de resultados
        st.session_state['agent_results'] = {
            "name": dataset_name,
            "format": "csv", # ou tsv, dependendo da origem
            "results_df": df_current_run
        }
        st.session_state.training_started = True
            
    # Only display training results if training has started
    if st.session_state.training_started:
        st.subheader("Resultados do Treinamento")
        from E_results.E1_results import results as show_agent_results # Import locally to avoid circular dependency issues
        show_agent_results() 
