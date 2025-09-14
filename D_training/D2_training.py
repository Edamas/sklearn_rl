import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import importlib
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from D_training.data_loader import load_parameters

def create_estimator_instance(class_path, params):
    """
    Dynamically imports a class and creates an instance with given parameters.
    """
    try:
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        estimator_class = getattr(module, class_name)
        instance = estimator_class(**params)
        return instance, None
    except Exception as e:
        return None, str(e)

def generate_random_params(estimator_name, params_df):
    """
    Generates a dictionary of random parameters for a given estimator,
    using the logical types and constraints from the parameters dataframe.
    """
    estimator_params_df = params_df[
        params_df['estimators_list'].apply(
            lambda x: estimator_name in x if isinstance(x, list) else False
        )
    ]
    
    params = {}

    for _, row in estimator_params_df.iterrows():
        param_name = row['param_name']
        param_dtype = row['param_dtype']
        param_list = row['param_list']
        param_min = row['param_min']
        param_max = row['param_max']

        if param_name in ['random_state', 'n_jobs']:
            continue

        value = None
        choice = None  # garante que 'choice' sempre existe
        if isinstance(param_list, list) and param_list:
            # Filter out complex types we can't handle yet
            simple_choices = [c for c in param_list if not isinstance(c, str) or c.lower() not in ['array', 'callable', 'float', 'int']]
            if simple_choices:
                choice = random.choice(simple_choices)
                # Convert string representations of booleans/None
                if isinstance(choice, str):
                    if choice.endswith('()') and '.' in choice:  # Looks like a class instantiation
                        try:
                            module_path, class_name = choice.rsplit('.', 1)
                            module = importlib.import_module(module_path)
                            cls = getattr(module, class_name[:-2])  # Remove '()'
                            value = cls()
                        except Exception as e:
                            st.warning(f"Could not instantiate {choice}: {e}. Using default or skipping.")
                            value = None
                    elif choice.lower() == 'none':
                        value = None
                    elif choice.lower() == 'true':
                        value = True
                    elif choice.lower() == 'false':
                        value = False
                    else:
                        value = choice
                else:
                    value = choice
            else:
                value = None

        # If no value from list, use dtype and range
        if value is None and not (isinstance(param_list, list) and None in param_list):
            if 'int' in str(param_dtype) and pd.notna(param_min) and pd.notna(param_max):
                value = random.randint(int(float(param_min)), int(float(param_max)))
            elif 'float' in str(param_dtype) and pd.notna(param_min) and pd.notna(param_max):
                value = random.uniform(float(param_min), float(param_max))
            elif 'bool' in str(param_dtype):
                value = random.choice([True, False])
        
        if value is not None:
            params[param_name] = value

    # Add common params
    if 'random_state' in estimator_params_df['param_name'].values:
        params['random_state'] = 42
    if 'n_jobs' in estimator_params_df['param_name'].values:
        params['n_jobs'] = -1
        
    return params

def generate_preprocessing_steps(group_name):
    """Dynamically generates a list of preprocessing steps (as dicts) for a given group type."""
    steps_repr = []
    
    imputers = {
        'Numeric': {
            'sklearn.impute.SimpleImputer': {'strategy': ['mean', 'median']},
            'sklearn.impute.KNNImputer': {'n_neighbors': [3, 4, 5, 6, 7, 8]}
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

    st.subheader("2.4 Treinamento do Agente")

    df = st.session_state.get("original_df")
    y_cols = st.session_state.get("y_cols")
    task_type = st.session_state.get("task_type")
    column_summary = st.session_state.get("column_summary_df")
    compatible_estimators = st.session_state.get("compatible_estimators")
    num_episodes = st.session_state.get("num_episodes")
    dataset_name = st.session_state.get("dataset_name")

    if any(arg is None for arg in [df, task_type, column_summary, compatible_estimators, num_episodes, dataset_name]):
        st.error("Dados de configuração essenciais estão faltando. Revise as etapas.")
        st.stop()
    
    # Ensure num_episodes is an int for Pylance
    num_episodes = int(num_episodes)

    if df is not None: # Explicitly check df is not None
        X = df.drop(columns=y_cols) if y_cols else df
        y = df[y_cols] if y_cols else None
    else:
        # This else block should ideally not be reached due to the st.stop() above,
        # but it satisfies Pylance that X and y are always defined.
        X = pd.DataFrame() # Or some other appropriate default
        y = None

    if 'training_started' not in st.session_state:
        st.session_state.training_started = False

    if st.button("🚀 Iniciar Treinamento", key="start_training_button"):
        params_df = load_parameters()

        X_train, y_train, scoring_metric = None, None, None # Initialize to None to satisfy Pylance

        if y is not None:
            if y_cols and len(y_cols) == 1:
                y_raveled = y.values.ravel()
            else:
                y_raveled = y.values
            X_train, _, y_train, _ = train_test_split(X, y_raveled, test_size=0.3, random_state=42)
            scoring_metric = "accuracy" if task_type == "Classification" else "r2"
        else:
            X_train, y_train, scoring_metric = X, None, 'adjusted_rand_score'

        all_trials_results = []
        progress_area = st.container()
        overall_progress_bar = progress_area.progress(0, text="Progresso Geral da Avaliação de Modelos")

        for i in range(int(num_episodes)):
            start_time = time.time()
            
            transformers_for_construction = []
            transformers_for_representation = []

            if column_summary is not None:
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
                    
                    if pipeline_steps_objects:
                        group_pipeline = Pipeline(pipeline_steps_objects)
                        transformers_for_construction.append((group_name, group_pipeline, feature_cols_in_group))
                        transformers_for_representation.append({
                            'group': group_name, 
                            'columns': feature_cols_in_group, 
                            'steps': steps_repr
                        })

            preprocessor = ColumnTransformer(transformers_for_construction, remainder='drop') if transformers_for_construction else "passthrough"

            if compatible_estimators is None or compatible_estimators.empty:
                st.error("Nenhum estimador compatível encontrado.")
                return

            estimator_def = compatible_estimators.sample(n=1).iloc[0]
            model_name = estimator_def['estimator_name']
            model_class_path = estimator_def['class_path']
            
            params = generate_random_params(model_name, params_df)

            processed_params = {}
            n_samples = X_train.shape[0]
            n_features = X_train.shape[1]
            n_classes = len(np.unique(y_train)) if y_train is not None else None

            for param_name, param_value in params.items():
                if isinstance(param_value, str):
                    if param_value == 'n_samples':
                        processed_params[param_name] = n_samples
                    elif param_value == 'n_features':
                        processed_params[param_name] = n_features
                    elif param_value == 'n_classes' and n_classes is not None:
                        processed_params[param_name] = n_classes
                    else:
                        processed_params[param_name] = param_value
                else:
                    processed_params[param_name] = param_value
            
            model, error_msg = create_estimator_instance(model_class_path, processed_params)

            pipeline_repr = {
                'preprocessor': transformers_for_representation,
                'estimator': {'name': model_name, 'class_path': model_class_path, 'params': params}
            }

            score, status = np.nan, "Erro"
            if not error_msg and y_train is not None:
                try:
                    main_pipe = Pipeline([('preprocessor', preprocessor), ('estimator', model)])
                    scores = cross_val_score(main_pipe, X_train, y_train, cv=3, scoring=scoring_metric)
                    score = np.mean(scores)
                    status = "Sucesso"
                    error_msg = ""
                except Exception as e:
                    error_msg = str(e)

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
            overall_progress_bar.progress((i + 1) / max(1, int(num_episodes)), text=progress_text)

        if all_trials_results:
            df_current_run = pd.DataFrame(all_trials_results)
            st.session_state['agent_results'] = {"name": dataset_name, "results_df": df_current_run}
            st.session_state.training_started = True
            st.rerun()

    if st.session_state.training_started:
        from E_results.E1_results import results
        results()