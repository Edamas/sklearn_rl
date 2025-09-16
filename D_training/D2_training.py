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

from B_input_config.B2_feature_engineering import apply_transformations
from D_training.data_loader import load_parameters
from sklearn.metrics import make_scorer, r2_score

def custom_r2_scorer(y_true, y_pred):
    """
    Calcula o R² score, mas remove valores NaN ou infinitos das predições
    antes de calcular o score para evitar erros.
    """
    finite_mask = np.isfinite(y_pred)
    y_true_finite = y_true[finite_mask]
    y_pred_finite = y_pred[finite_mask]

    if len(y_true_finite) == 0:
        return np.nan  # Retorna NaN se não houver predições válidas

    return r2_score(y_true_finite, y_pred_finite)

def create_estimator_instance(class_path, params):
    """
    Cria dinamicamente uma instância de uma classe do scikit-learn a partir de seu
    caminho de importação e um dicionário de parâmetros.

    Retorna a instância e uma mensagem de erro, se houver.
    """
    try:
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        estimator_class = getattr(module, class_name)
        instance = estimator_class(**params)
        return instance, None
    except Exception as e:
        return None, str(e)

def _get_random_value_for_param(row, n_samples, n_features, n_classes):
    param_name = row['param_name']
    param_dtype = row['param_dtype']
    param_list = row['param_list']
    param_min = row['param_min']
    param_max = row['param_max']

    value = None

    # Handle special cases first
    if param_name == 'n_estimators':
        if pd.notna(param_min) and pd.notna(param_max):
            return random.randint(int(param_min), int(param_max))
        else:
            return 100 # Default value

    # Handle param_list (categorical/list choices)
    if isinstance(param_list, list) and param_list:
        # Filter out 'array' and 'callable' if they are just placeholders
        simple_choices = [c for c in param_list if not (isinstance(c, str) and c.lower() in ['array', 'callable'])]
        
        if simple_choices:
            choice = random.choice(simple_choices)
            
            # Handle special string values
            if isinstance(choice, str):
                if choice.endswith('()') and '.' in choice: # Callable/Class instantiation
                    try:
                        module_path, class_name_str = choice.rsplit('.', 1)
                        class_name = class_name_str[:-2]
                        module = importlib.import_module(module_path)
                        cls = getattr(module, class_name)
                        return cls()
                    except Exception:
                        return None # Failed to instantiate
                elif choice.lower() == 'none':
                    return None
                elif choice.lower() == 'true':
                    return True
                elif choice.lower() == 'false':
                    return False
                elif choice == 'n_samples' and n_samples is not None:
                    return n_samples
                elif choice == 'n_features' and n_features is not None:
                    return n_features
                elif choice == 'n_classes' and n_classes is not None:
                    return n_classes
                else:
                    # Special handling for class_weight if it's a string
                    if param_name == 'class_weight':
                        if choice.lower() == 'balanced':
                            return 'balanced'
                        elif choice.lower() == 'none':
                            return None
                        # If it's a string but not 'balanced' or 'none', let it pass, sklearn will validate
                        return choice
                    return choice # Regular string or other literal
            else:
                # If choice is not a string, it's a literal (e.g., number, dict, tuple)
                # Special handling for class_weight if it's a dict
                if param_name == 'class_weight':
                    # If it's a dict, and it's not 'balanced' or None, it's likely invalid for random generation.
                    # Convert to None to avoid the error, as we cannot dynamically generate a valid dict.
                    return None
                return choice

    # Handle numeric and boolean types if no param_list or param_list was empty/filtered
    if 'int' in str(param_dtype) and pd.notna(param_min) and pd.notna(param_max):
        return random.randint(int(param_min), int(param_max))
    elif 'float' in str(param_dtype) and pd.notna(param_min) and pd.notna(param_max):
        return random.uniform(float(param_min), float(param_max))
    elif 'bool' in str(param_dtype):
        return random.choice([True, False])
    
    return value # Return None if no suitable value was generated

def generate_random_params(estimator_name, params_df, n_samples=None, n_features=None, n_classes=None):
    """
    Gera um dicionário de parâmetros aleatórios para um dado estimador.
    A função utiliza os tipos de dados e restrições do `parameters.tsv`.

    Atenção: Esta função é crucial e sensível. Erros na geração de parâmetros
    podem causar falhas inesperadas nos estimadores do scikit-learn.
    """
    estimator_params_df = params_df[
        params_df['estimators_list'].apply(
            lambda x: estimator_name in x if isinstance(x, list) else False
        )
    ]
    
    params = {}

    for _, row in estimator_params_df.iterrows():
        param_name = row['param_name']

        if param_name in ['random_state', 'n_jobs']:
            continue

        try:
            value = _get_random_value_for_param(row, n_samples, n_features, n_classes)
            if value is not None:
                params[param_name] = value
        except Exception as e:
            st.warning(f"Falha ao gerar parâmetro '{param_name}' para '{estimator_name}': {e}")

    # Adiciona parâmetros comuns com valores fixos
    if 'random_state' in estimator_params_df['param_name'].values:
        params['random_state'] = 42
    if 'n_jobs' in estimator_params_df['param_name'].values:
        params['n_jobs'] = -1
        
    return params

def generate_preprocessing_steps(group_name):
    """
    Gera dinamicamente uma lista de etapas de pré-processamento para um grupo de features.
    Retorna uma lista de dicionários, cada um representando uma etapa.
    """
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
    """
    Função principal para o Passo 7: Treinamento do Agente.
    Esta função é chamada depois que o usuário configura e seleciona os estimadores.
    Ela executa o loop de treinamento para o número de episódios especificado.
    """
    if st.session_state.get("y_cols") is None or st.session_state.get("num_episodes") is None:
        st.warning("Nenhum resultado de agente encontrado. Execute o agente primeiro.") # Moved this message here
        return

    st.markdown("### Treinamento do Agente")

    # Carrega todas as informações necessárias do estado da sessão
    df = st.session_state.get("processed_df") # Use the processed_df
    config_df = st.session_state.get("feature_config_df")
    task_type = st.session_state.get("task_type")
    column_summary = st.session_state.get("column_summary_df") # This line should be removed later
    compatible_estimators = st.session_state.get("compatible_estimators")
    num_episodes = st.session_state.get("num_episodes")
    dataset_name = st.session_state.get("dataset_name")

    if any(arg is None for arg in [df, config_df, task_type, compatible_estimators, num_episodes, dataset_name]):
        st.error("Dados de configuração essenciais estão faltando. Revise as etapas anteriores.")
        st.stop()
    
    if num_episodes is None:
        st.error("Número de episódios não definido. Por favor, configure o agente.")
        st.stop()
    num_episodes = int(num_episodes)

    # Aplica as transformações de feature engineering definidas pelo usuário
    processed_df, X_cols_transformed, y_cols_transformed = apply_transformations(df, config_df)
    st.session_state['processed_df_for_current_task'] = processed_df

    X = processed_df[X_cols_transformed]
    y = processed_df[y_cols_transformed] if y_cols_transformed else None

    if 'training_started' not in st.session_state:
        st.session_state.training_started = False

    if st.button("🚀 Iniciar Treinamento", key="start_training_button"):
        params_df = load_parameters()

        # Prepara os dados de treino e teste
        if y is not None:
            y_raveled = y.values.ravel() if len(y_cols_transformed) == 1 else y.values
            X_train, _, y_train, _ = train_test_split(X, y_raveled, test_size=0.3, random_state=42)
        else:
            X_train, y_train = X, None

        # Remove NaNs dos dados de treino para evitar erros nos estimadores
        if y_train is not None:
            # X_train should already be a DataFrame from train_test_split if X was a DataFrame
            # We need to ensure it remains a DataFrame after NaN handling
            temp_df = X_train.copy()
            temp_df['target'] = y_train
            temp_df.dropna(inplace=True)
            X_train = temp_df.drop(columns=['target']) # Keep X_train as DataFrame
            y_train = temp_df['target'].values # y_train can be a NumPy array
        else:
            # X_train should already be a DataFrame
            X_train = X_train.dropna() # Keep X_train as DataFrame

        all_trials_results = []
        progress_area = st.container()
        overall_progress_bar = progress_area.progress(0, text="Progresso Geral da Avaliação de Modelos")

        # Initialize all_predictions_df with X and y from the processed_df
        # Ensure processed_df has the original index
        base_df_for_predictions = processed_df.copy()
        if y_cols_transformed and y is not None:
            # If y is a Series or 1D array, ensure it's added as a column
            if isinstance(y, pd.Series):
                base_df_for_predictions[y_cols_transformed[0]] = y
            elif isinstance(y, np.ndarray) and y.ndim == 1:
                base_df_for_predictions[y_cols_transformed[0]] = y
            elif isinstance(y, pd.DataFrame):
                for col in y.columns:
                    base_df_for_predictions[col] = y[col]
            else:
                st.warning("Tipo de 'y' inesperado. As colunas alvo podem não ser adicionadas corretamente ao base_df_for_predictions.")
        
        # Ensure X_cols are present in base_df_for_predictions
        for col in X_cols_transformed:
            if col not in base_df_for_predictions.columns:
                base_df_for_predictions[col] = processed_df[col] # Add if missing

        all_predictions_df = base_df_for_predictions.copy()

        # Loop principal de treinamento (cada iteração é um "episódio")
        for i in range(int(num_episodes)):
            start_time = time.time()
            
            transformers_for_construction = []
            transformers_for_representation = []

            # 1. Monta o pipeline de pré-processamento dinamicamente
            # Ensure config_df is valid and has necessary columns
            if config_df is None or config_df.empty or 'selected_type' not in config_df.columns or 'is_feature' not in config_df.columns or 'column_name' not in config_df.columns:
                st.error("Erro: Configuração de features inválida ou incompleta. Por favor, revise a Engenharia de Features.")
                return # Stop agent_training if config_df is not valid

            # Identify numeric and categorical features based on config_df
            # Filter to ensure only columns present in X are considered for preprocessing
            all_X_cols_in_current_df = X.columns.tolist()

            numeric_cols = config_df[(config_df['selected_type'] == 'Numeric') & (config_df['is_feature'] == True)]['column_name'].tolist()
            numeric_cols = [col for col in numeric_cols if col in all_X_cols_in_current_df]

            categorical_cols = config_df[(config_df['selected_type'] == 'Categorical') & (config_df['is_feature'] == True)]['column_name'].tolist()
            categorical_cols = [col for col in categorical_cols if col in all_X_cols_in_current_df]

            transformers_for_construction = []
            transformers_for_representation = []

            # Build preprocessor for Numeric features
            if numeric_cols:
                numeric_steps_repr = generate_preprocessing_steps('Numeric')
                numeric_pipeline_steps_objects = []
                for step_name, step_info in numeric_steps_repr:
                    step_obj, _ = create_estimator_instance(step_info['class_path'], step_info['params'])
                    if step_obj:
                        numeric_pipeline_steps_objects.append((step_name, step_obj))
                
                if numeric_pipeline_steps_objects:
                    numeric_pipeline = Pipeline(numeric_pipeline_steps_objects)
                    transformers_for_construction.append(('numeric_preprocessor', numeric_pipeline, numeric_cols))
                    transformers_for_representation.append({
                        'group': 'Numeric', 
                        'columns': numeric_cols, 
                        'steps': numeric_steps_repr
                    })

            # Build preprocessor for Categorical features
            if categorical_cols:
                categorical_steps_repr = generate_preprocessing_steps('Categorical')
                categorical_pipeline_steps_objects = []
                for step_name, step_info in categorical_steps_repr:
                    step_obj, _ = create_estimator_instance(step_info['class_path'], step_info['params'])
                    if step_obj:
                        categorical_pipeline_steps_objects.append((step_name, step_obj))
                
                if categorical_pipeline_steps_objects:
                    categorical_pipeline = Pipeline(categorical_pipeline_steps_objects)
                    transformers_for_construction.append(('categorical_preprocessor', categorical_pipeline, categorical_cols))
                    transformers_for_representation.append({
                        'group': 'Categorical', 
                        'columns': categorical_cols, 
                        'steps': categorical_steps_repr
                    })

            preprocessor = ColumnTransformer(transformers_for_construction, remainder='drop') if transformers_for_construction else "passthrough"

            if compatible_estimators is None or compatible_estimators.empty:
                st.warning("Nenhum estimador compatível encontrado para o treinamento.")
                return

            # 2. Seleciona um estimador aleatório da lista de compatíveis
            estimator_def = compatible_estimators.sample(n=1).iloc[0]
            model_name = estimator_def['estimator_name']
            model_class_path = estimator_def['class_path']
            
            # 3. Gera parâmetros aleatórios para o estimador selecionado
            n_samples = X_train.shape[0]
            n_features = X_train.shape[1]
            if y_train is not None and isinstance(y_train, np.ndarray):
                n_classes = len(np.unique(y_train))
            else:
                n_classes = None

            params = generate_random_params(model_name, params_df, n_samples, n_features, n_classes)
            
            # 4. Define a métrica de scoring com base no tipo de tarefa
            if task_type == "Regression":
                scorer_to_use = make_scorer(custom_r2_scorer)
            else:
                scorer_to_use = "accuracy"

            # 5. Cria a instância do estimador e o pipeline completo
            estimator_instance, error = create_estimator_instance(model_class_path, params)
            
            pipeline = None
            if estimator_instance is not None and hasattr(estimator_instance, 'fit'): # Check if it's a valid estimator
                pipeline = Pipeline([('preprocessor', preprocessor), ('estimator', estimator_instance)])
            else:
                error_message = error if error else f"Estimador {model_name} não pôde ser instanciado ou não é um estimador válido."
                status = "Erro"
                score = np.nan

            score = np.nan
            status = "Erro"
            error_message = ""
            # processed_df_with_preds = None # Initialize to None - NO LONGER NEEDED PER TRIAL

            # 6. Executa o treinamento e a validação cruzada
            if pipeline:
                try:
                    scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring=scorer_to_use, error_score='raise')
                    score = np.mean(scores)
                    status = "Sucesso"

                    # Fit the pipeline on the full training data before making predictions
                    pipeline.fit(X_train, y_train)

                    # Make predictions on the full processed_df
                    # Use X_full from the original processed_df to ensure consistent indexing
                    X_full_for_predict = processed_df[X_cols_transformed]

                    if hasattr(pipeline, "predict"):
                        preds = pipeline.predict(X_full_for_predict)
                    else:
                        # Handle cases where pipeline might not have direct predict (e.g., only transform)
                        if hasattr(pipeline, "named_steps") and 'estimator' in pipeline.named_steps:
                            core_est = pipeline.named_steps['estimator']
                            if 'preprocessor' in pipeline.named_steps:
                                transformed = pipeline.named_steps['preprocessor'].transform(X_full_for_predict)
                                preds = core_est.predict(transformed)
                            else:
                                preds = core_est.predict(X_full_for_predict)
                        else:
                            preds = np.full(len(X_full_for_predict), np.nan) # No predict method, fill with NaN

                    # Add predictions to all_predictions_df with episode index as column name
                    all_predictions_df[str(i)] = np.array(preds)

                except Exception as e:
                    error_message = str(e)
                    score = np.nan
                    status = "Erro"
                    # If an error occurs, fill the prediction column for this episode with NaN
                    all_predictions_df[str(i)] = np.nan 
            else:
                error_message = error
                all_predictions_df[str(i)] = np.nan # Pipeline not created, fill with NaN

            end_time = time.time()
            duration = end_time - start_time

            # 7. Armazena os resultados do episódio
            trial_result = {
                "timestamp": datetime.now(),
                "estimator_name": model_name,
                "status": status,
                "score": score,
                "duration_seconds": duration,
                "error": error_message,
                "pipeline_steps": {
                    "preprocessor": transformers_for_representation,
                    "estimator": {"class_path": model_class_path, "params": params}
                },
                "fitted_pipeline_obj": pipeline if status == "Sucesso" else None,
                "training_X_cols": X_cols_transformed,
                "training_y_cols": y_cols_transformed,
                "target_column": y_cols_transformed[0] if y_cols_transformed and len(y_cols_transformed) == 1 else None,
                "episode_index": i # Store the episode index
            }
            all_trials_results.append(trial_result)

            # Atualiza a interface do usuário
            progress = (i + 1) / num_episodes
            overall_progress_bar.progress(progress, text=f"Episódio {i+1}/{num_episodes}: {model_name} | Score: {score:.4f}" if status == "Sucesso" else f"Episódio {i+1}/{num_episodes}: {model_name} | Erro")

        # 8. Ao final dos episódios, armazena os resultados e reinicia a página para exibição
        if all_trials_results:
            df_current_run = pd.DataFrame(all_trials_results)
            st.session_state['agent_results'] = {
                "name": dataset_name,
                "results_df": df_current_run,
                "all_predictions_df": all_predictions_df, # Store the comprehensive predictions DataFrame
                "X_cols_transformed": X_cols_transformed,
                "y_cols_transformed": y_cols_transformed
            }
            st.session_state.training_started = True
            st.rerun()

    # Se o treinamento já ocorreu, ou se o botão de treinamento foi clicado, exibe a página de resultados
    if st.session_state.training_started or st.session_state.get("start_training_button"):
        from E_results.E1_results import results
        results()

            