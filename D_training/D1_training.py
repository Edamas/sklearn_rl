import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from D_training.agent_rl import AgentHyperparameterParser, create_estimator_instance
from functions import log_message

PARAMETERS_FILE = st.session_state.files.get('parameters')

def generate_preprocessing_steps(group_name):
    """Dynamically generates a list of preprocessing steps (as dicts) for a given group type."""
    steps_repr = []
    
    imputers = {
        'Numeric': {
            'sklearn.impute.SimpleImputer': {'strategy': ['mean', 'median']},
            'sklearn.impute.KNNImputer': {'n_neighbors': [3, 5, 7]}
        },
        'Categorical': {
            'sklearn.impute.SimpleImputer': {'strategy': ['most_frequent', 'constant']}
        }
    }
    scalers = {
        'Numeric': {
            'sklearn.preprocessing.StandardScaler': {},
            'sklearn.preprocessing.MinMaxScaler': {},
            'sklearn.preprocessing.RobustScaler': {}
        }
    }
    encoders = {
        'Categorical': {
            'sklearn.preprocessing.OneHotEncoder': {'handle_unknown': ['ignore']}
        }
    }

    if group_name in imputers:
        class_path, params_space = random.choice(list(imputers[group_name].items()))
        params = {k: random.choice(v) for k, v in params_space.items()}
        steps_repr.append(('imputer', {'class_path': class_path, 'params': params}))

    if group_name in scalers and random.random() > 0.25:
        class_path, params_space = random.choice(list(scalers[group_name].items()))
        params = {k: random.choice(v) for k, v in params_space.items()}
        steps_repr.append(('scaler', {'class_path': class_path, 'params': params}))

    if group_name in encoders:
        class_path, params_space = random.choice(list(encoders[group_name].items()))
        params = {k: random.choice(v) for k, v in params_space.items()}
        steps_repr.append(('encoder', {'class_path': class_path, 'params': params}))

    if not steps_repr:
        steps_repr.append(('default_imputer', {
            'class_path': 'sklearn.impute.SimpleImputer',
            'params': {'strategy': 'most_frequent'}
        }))

    return steps_repr

def agent_training():
    if st.session_state.get("y_cols") is None or st.session_state.get("num_episodes") is None:
        return

    st.subheader("4. Treinamento do Agente")

    df = st.session_state.get("original_df")
    y_cols = st.session_state.get("y_cols")
    task_type = st.session_state.get("task_type")
    column_summary = st.session_state.get("column_summary_df")
    compatible_estimators = st.session_state.get("compatible_estimators")
    num_episodes = st.session_state.get("num_episodes")
    dataset_name = st.session_state.get("dataset_name")

    if any(arg is None for arg in [df, task_type, column_summary, compatible_estimators, num_episodes, dataset_name]):
        log_message("WARNING", "Dados de configuração essenciais estão faltando. Revise as etapas.")
        st.stop()

    if 'training_started' not in st.session_state:
        st.session_state.training_started = False

    if st.button("🚀 Iniciar Treinamento", key="start_training_button"):
        X = df.drop(columns=y_cols) if y_cols else df
        y = df[y_cols] if y_cols else None

        try:
            params_df = pd.read_csv(PARAMETERS_FILE, sep='\t')
        except FileNotFoundError as e:
            log_message("EXCEPTION", f"Arquivo de parâmetros '{PARAMETERS_FILE}' não encontrado.", exception=e)
            return

        if y is not None:
            y_raveled = y.values.ravel() if len(y_cols) == 1 else y.values
            X_train, _, y_train, _ = train_test_split(X, y_raveled, test_size=0.3, random_state=42)
            scoring_metric = "accuracy" if task_type == "Classification" else "r2"
        else:
            X_train, y_train, scoring_metric = X, None, 'adjusted_rand_score'

        parser = AgentHyperparameterParser(compatible_estimators, params_df)
        all_trials_results = []
        progress_area = st.container()
        overall_progress_bar = progress_area.progress(0, text="Progresso Geral da Avaliação de Modelos")

        for i in range(num_episodes):
            start_time = time.time()
            
            transformers_for_construction = []
            transformers_for_representation = []

            for group_name, group_info in column_summary.iterrows():
                feature_cols_in_group = [col for col in group_info['columns'] if col in X.columns]
                if not feature_cols_in_group:
                    continue
                
                steps_repr = generate_preprocessing_steps(group_name)
                
                pipeline_steps_objects = []
                for step_name, step_info in steps_repr:
                    step_obj, _ = create_estimator_instance(step_info['class_path'], step_info['params'])
                    if step_obj:
                        pipeline_steps_objects.append((step_name, step_obj))
                
                group_pipeline = Pipeline(pipeline_steps_objects)
                transformers_for_construction.append((group_name, group_pipeline, feature_cols_in_group))
                transformers_for_representation.append({
                    'group': group_name, 
                    'columns': feature_cols_in_group, 
                    'steps': steps_repr
                })

            preprocessor = ColumnTransformer(transformers_for_construction, remainder='drop')

            estimator_def = compatible_estimators.sample(n=1).iloc[0]
            model_name = estimator_def['estimator_name']
            model_class_path = estimator_def['class_path']
            params = parser.generate_random_params(estimator_def, estimator_name=model_name)
            model, error_msg = create_estimator_instance(model_class_path, params)

            pipeline_repr = {
                'preprocessor': transformers_for_representation,
                'estimator': {'name': model_name, 'class_path': model_class_path, 'params': params}
            }

            score, status = np.nan, "Erro"
            if not error_msg:
                main_pipe = Pipeline([('preprocessor', preprocessor), ('estimator', model)])
                try:
                    scores = cross_val_score(main_pipe, X_train, y_train, cv=3, scoring=scoring_metric)
                    score = np.mean(scores)
                    status = "Sucesso"
                except Exception as e:
                    error_msg = str(e).replace('\n', ' ')
                    log_message("EXCEPTION", f"Erro durante cross_val_score para {model_name}.", exception=e, display_streamlit=False)

            end_time = time.time()
            duration = end_time - start_time
            
            all_trials_results.append({
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': duration,
                'estimator_name': model_name,
                'params': params,
                'status': status,
                'score': score if not np.isnan(score) else 0,
                'error': error_msg if error_msg else '',
                'pipeline_steps': pipeline_repr
            })

            progress_text = f"Episódio: {i+1}/{num_episodes} | Modelo: **{model_name}** | Score: {score:.3f}"
            overall_progress_bar.progress((i + 1) / num_episodes, text=progress_text)

        if all_trials_results:
            df_current_run = pd.DataFrame(all_trials_results)
            st.session_state['agent_results'] = {"name": dataset_name, "results_df": df_current_run}
        
        st.session_state.training_started = True
        st.rerun()

    if st.session_state.training_started:
        from E_results.E1_results import results
        results()