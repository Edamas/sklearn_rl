import streamlit as st
import pandas as pd
import numpy as np
import random
import pydoc
import time
import ast
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import plotly.express as px

# --- Constantes de Arquivos ---
HISTORY_FILE = st.session_state.files.get('log')
ESTIMATORS_FILE = st.session_state.files.get('estimators')
PARAMETERS_FILE = st.session_state.files.get('parameters')

class AgentHyperparameterParser:
    """Lida com a geração de hiperparâmetros baseada nos novos arquivos de configuração."""
    def __init__(self, compatible_estimators_df, params_df):
        self.estimators_df = compatible_estimators_df
        self.all_params_df = params_df.set_index('param_name') # Mantém todos os params para referência
        # Garante que a coluna booleana seja do tipo bool e filtra para treinamento
        params_df['apt_for_training'] = params_df['apt_for_training'].astype(bool)
        self.params_df = params_df[params_df['apt_for_training']].set_index('param_name')

    

    def generate_random_params(self, estimator_series, estimator_name: str):
        """Gera um dicionário de parâmetros aleatórios para um dado estimador."""
        params = {}
        param_names_str = estimator_series.get('params_list', '[]')
        
        try:
            # Use ast.literal_eval for safe parsing of the list string
            param_names = ast.literal_eval(param_names_str)
            if not isinstance(param_names, list):
                param_names = []
        except (ValueError, SyntaxError):
            param_names = []

        for param_name in param_names:
            # Verifica se o parâmetro existe no arquivo de parâmetros
            if param_name not in self.all_params_df.index:
                st.warning(f"Aviso: Parâmetro '{param_name}' não encontrado em '{PARAMETERS_FILE}'. Pulando.")
                continue

            # Verifica se o parâmetro está apto para treinamento antes de gerar valor
            if param_name not in self.params_df.index:
                continue # Pula silenciosamente os parâmetros não aptos

            # --- Nova lógica para selecionar a regra correta ---
            candidate_rules = self.params_df.loc[param_name]
            
            selected_rule = None
            if isinstance(candidate_rules, pd.DataFrame):
                # Tenta encontrar uma regra específica para o estimador atual
                for _, rule in candidate_rules.iterrows():
                    estimators_list_str = rule.get('estimators_list', '[]')
                    try:
                        # Safely evaluate the string representation of the list
                        rule_estimators = ast.literal_eval(estimators_list_str)
                        if not isinstance(rule_estimators, list):
                            rule_estimators = []
                    except (ValueError, SyntaxError):
                        rule_estimators = []

                    if estimator_name in rule_estimators:
                        selected_rule = rule
                        break # Encontrou a regra específica, pode parar
                
                # Se não encontrou uma regra específica, procura uma regra geral (lista vazia)
                if selected_rule is None:
                    for _, rule in candidate_rules.iterrows():
                        estimators_list_str = rule.get('estimators_list', '[]')
                        try:
                            rule_estimators = ast.literal_eval(estimators_list_str)
                            if not isinstance(rule_estimators, list):
                                rule_estimators = []
                        except (ValueError, SyntaxError):
                            rule_estimators = []
                        
                        if not rule_estimators: # Se a lista de estimadores está vazia, é uma regra geral
                            selected_rule = rule
                            break # Usa a primeira regra geral encontrada
            else: # Apenas uma regra para este parâmetro
                selected_rule = candidate_rules
            
            if selected_rule is None:
                st.warning(f"Aviso: Nenhuma regra de parâmetro adequada encontrada para '{param_name}' e estimador '{estimator_name}'. Pulando.")
                continue
            
            param_rules = selected_rule # Usa a regra selecionada
            # --- Fim da nova lógica ---

            param_type = param_rules.get('param_dtype')
            
            # Lida com valores que podem ser None
            param_list_str = param_rules.get('param_list', '')
            can_be_none = '[none]' in str(param_list_str)
            if can_be_none and random.choice([True, False]):
                params[param_name] = None
                continue

            try:
                if param_type == 'int' or param_type == 'float':
                    try:
                        default = float(param_rules['param_standard'])
                        min_val = float(param_rules['param_min'])
                        max_val = float(param_rules['param_max'])
                        
                        # Use a normal distribution centered around the default
                        sigma = (max_val - min_val) / 4  # Std dev is 1/4 of the range
                        if sigma == 0:
                            value = default
                        else:
                            value = np.random.normal(default, sigma)
                        
                        # Clip the value to be within the min/max bounds
                        value = np.clip(value, min_val, max_val)
                        
                        if param_type == 'int':
                             params[param_name] = int(round(value))
                        else: # float
                             params[param_name] = value

                    except (ValueError, TypeError):
                        # Fallback to uniform for non-numeric defaults or invalid min/max
                        try:
                            if param_type == 'int':
                                min_val = int(param_rules['param_min'])
                                max_val = int(param_rules['param_max'])
                                params[param_name] = random.randint(min_val, max_val)
                            else: # float
                                min_val = float(param_rules['param_min'])
                                max_val = float(param_rules['param_max'])
                                params[param_name] = random.uniform(min_val, max_val)
                        except (ValueError, TypeError):
                             pass # Silently skip if min/max are also invalid

                elif param_type == 'cat':
                    values_str = param_rules.get('param_list', '[]')
                    try:
                        raw_values = ast.literal_eval(values_str)
                        if not isinstance(raw_values, list) or not raw_values:
                            continue
                        
                        # Convert string booleans/none to actual objects, leave other types as is
                        possible_values = []
                        for v in raw_values:
                            if isinstance(v, str):
                                v_lower = v.lower()
                                if v_lower == 'true':
                                    possible_values.append(True)
                                elif v_lower == 'false':
                                    possible_values.append(False)
                                elif v_lower == 'none':
                                    possible_values.append(None)
                                else:
                                    possible_values.append(v)
                            else:
                                possible_values.append(v)

                    except (ValueError, SyntaxError):
                        continue
                    
                    # Filter out None if it was already handled by the can_be_none logic
                    if can_be_none:
                        possible_values = [v for v in possible_values if v is not None]

                    if not possible_values:
                        continue

                    params[param_name] = random.choice(possible_values)
                elif param_type == 'bool':
                    params[param_name] = random.choice([True, False])

            except (ValueError, IndexError, TypeError) as e:
                st.warning(f"Aviso: pulando parâmetro '{param_name}' devido a regra malformada: {param_rules.to_dict()} -> {e}")
        
        return params

    def get_unique_random_estimators(self, num_models_to_evaluate: int):
        """
        Seleciona um número especificado de estimadores únicos e aleatórios
        da lista de estimadores compatíveis.
        """
        if self.estimators_df.empty:
            return pd.DataFrame() # Retorna um DataFrame vazio se não houver estimadores

        # Garante que não tentamos selecionar mais estimadores do que disponíveis
        num_to_select = min(num_models_to_evaluate, len(self.estimators_df))
        
        # Seleciona estimadores aleatoriamente
        # .sample(frac=1) embaralha o DataFrame
        # .head(num_to_select)
        # ou .sample(n=num_to_select) se num_to_select < len(self.estimators_df)
        # Para garantir unicidade e aleatoriedade, sample(n=...) é mais direto
        
        if num_to_select == 0:
            return pd.DataFrame()

        # Se num_to_select for igual ao número total de estimadores, sample(frac=1) é mais eficiente
        if num_to_select == len(self.estimators_df):
            return self.estimators_df.sample(frac=1, random_state=42) # Usar random_state para reprodutibilidade
        else:
            return self.estimators_df.sample(n=num_to_select, random_state=42)

def create_estimator_instance(class_path, params):
    """Cria uma instância de um estimador de forma segura."""
    try:
        estimator_class = pydoc.locate(class_path)
        if estimator_class is None or not isinstance(estimator_class, type):
            return None, f"Classe não encontrada ou inválida: {class_path}"
        return estimator_class(**params), None
    except Exception as e:
        return None, str(e)

def run_agent():
    st.subheader("4. Execução e Resultados do Agente")
    agent_data = st.session_state.get("agent_data")
    if not agent_data or agent_data.get("X") is None:
        st.error("Erro interno: dados do agente não encontrados no session_state.")
        return

    # --- Carrega dados e configurações ---
    try:
        params_df = pd.read_csv(PARAMETERS_FILE, sep='\t')
    except FileNotFoundError:
        st.error(f"Arquivo de parâmetros '{PARAMETERS_FILE}' não encontrado.")
        return

    X = agent_data["X"]
    y = agent_data["y"]
    dataset_name = agent_data.get("name", "unknown_dataset")
    dataset_summary = agent_data.get("summary", "{}")
    compatible_estimators = agent_data.get("compatible_estimators")
    agent_config = agent_data.get("agent_config")

    if compatible_estimators is None or agent_config is None:
        st.error("Erro interno: configuração do agente não encontrada no session_state.")
        return

    num_models_to_evaluate = agent_config["num_models_to_evaluate"]
    num_param_optimization_trials = agent_config["num_param_optimization_trials"]
    
    # Trata o caso de y ser None (não supervisionado)
    if y is not None:
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
        classification_task = (not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20)
        scoring_metric = "accuracy" if classification_task else "r2"
    else:
        X_train, _ = train_test_split(X, test_size=0.3, random_state=42)
        y_train = None
        classification_task = False
        scoring_metric = None # Usa o score padrão do estimador para clusterização/etc.

    # --- Inicia o processo de otimização ---
    parser = AgentHyperparameterParser(compatible_estimators, params_df)
    estimators_to_evaluate = parser.get_unique_random_estimators(num_models_to_evaluate)

    if estimators_to_evaluate.empty:
        st.error("Erro: Nenhum estimador compatível encontrado para avaliação. Ajuste a seleção de features/alvos ou o arquivo de estimadores.")
        return

    all_trials_results = []
    progress_area = st.container()
    overall_progress_bar = progress_area.progress(0, text="Progresso Geral da Avaliação de Modelos")

    for i, (idx, estimator_def) in enumerate(estimators_to_evaluate.iterrows()):
        model_name = estimator_def['estimator_name']
        model_progress_text = progress_area.empty()
        inner_progress_bar = progress_area.progress(0)
        
        model_scores = []
        model_durations = []

        for j in range(num_param_optimization_trials):
            params = parser.generate_random_params(estimator_def, estimator_def['estimator_name'])
            
            start_time = time.time()
            model, error_msg = create_estimator_instance(estimator_def['class_path'], params)
            
            score = np.nan
            status = "Erro"
            if not error_msg:
                pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("model", model)]) # with_mean=False para matrizes esparsas
                try:
                    scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring=scoring_metric)
                    score = scores.mean()
                    status = "Sucesso"
                except Exception as e:
                    error_msg = str(e).replace('\n', ' ')
            
            end_time = time.time()
            duration = end_time - start_time
            if not np.isnan(score):
                model_scores.append(score)
                model_durations.append(duration)

            avg_score = np.mean(model_scores) if model_scores else 0
            avg_duration = np.mean(model_durations) if model_durations else 0

            trial_result = {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': duration,
                'estimator_name': model_name,
                'params': str(params),
                'score': score if not np.isnan(score) else 0,
                'error': error_msg if error_msg else ''
            }
            all_trials_results.append(trial_result)

            progress_text = f"Modelo: **{model_name}** | Tentativa {j+1}/{num_param_optimization_trials} | Média Score: {avg_score:.3f} | Média Tempo: {avg_duration:.2f}s"
            model_progress_text.write(progress_text)
            inner_progress_bar.progress((j + 1) / num_param_optimization_trials)
        
        overall_progress_bar.progress((i + 1) / len(estimators_to_evaluate), text="Progresso Geral da Avaliação de Modelos")

    if not all_trials_results:
        st.warning("Nenhum resultado foi gerado pelo agente.")
        return

    # Dataframe apenas com os resultados da execução atual
    df_current_run = pd.DataFrame(all_trials_results)

    # Salva os resultados no session_state para a página de resultados
    st.session_state['agent_results'] = {
        "name": dataset_name,
        "format": "csv", # ou tsv, dependendo da origem
        "results_df": df_current_run
    }
