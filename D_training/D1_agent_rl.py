

# import warnings
# # warnings.filterwarnings("ignore") 


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
import importlib # Added for dynamic imports <- (necessário?)
# from sklearn.experimental import enable_iterative_imputer # nào usado

estimators_file_path = st.session_state.files.get('estimators_optimized', 'D_training/estimators_for_algorithm.tsv')
parameters_file_path = st.session_state.files.get('parameters', 'D_training/parameters.tsv')
class AgentRL:
    def __init__(self, files_dict, project_root):
        self.project_root = project_root
        
        # Construct absolute paths using project_root and relative paths from files_dict
        # estimators_file_path = os.path.join(self.project_root, files_dict['sklearn_methods.tsv'])
        # parameters_file_path = os.path.join(self.project_root, files_dict['parameters.tsv'])
        # discutimos: Transformers, Clusters, Classification e Regression.
        self.estimators_df = pd.read_csv(estimators_file_path, sep='\t')  # lê estimators.tsv, com 266 estimadores (dos 4 tipos da tabela)
        self.parameters_df = pd.read_csv(parameters_file_path, sep='\t')  # lê parameters.tsv com 1860 parâmetros (definir random_state, n_jobs, etc.)
        self.pipeline = None
        self.model = None
        self.preprocessor = None
        self.target_column = None
        self.estimators_map = self._create_estimators_map()
        self.estimator_metadata = self._load_estimator_metadata()

    def _create_estimators_map(self):
        estimators_map = {}
        for _, row in self.estimators_df.iterrows():
            estimator_name = row['estimator_name']
            class_path = row['class_path']

            if pd.isna(class_path) or class_path == '0': 
                # Handle cases where class_path might be missing or '0'
                # For utility functions or mixins that are not meant to be instantiated directly
                # or if the class_path is not available, skip or handle as needed.
                # For now, we'll just skip.
                continue

            try:
                module_name, class_name = class_path.rsplit('.', 1)
                module = importlib.import_module(module_name)
                estimator_class = getattr(module, class_name)
                estimators_map[estimator_name] = estimator_class
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not import {estimator_name} from {class_path}. Error: {e}")
                # Optionally, log this error or add to a list of unavailable estimators
        return estimators_map

    def _load_estimator_metadata(self):
        metadata = {}
        for _, row in self.estimators_df.iterrows():
            estimator_name = row['estimator_name']
            metadata[estimator_name] = {
                'estimator_type': row['estimator_type'],
                'class_path': row['class_path'],
                'input_X_structure': row['input_X_structure'],
                'input_X_types': row['input_X_types'],
                'input_y_structure': row['input_y_structure'],
                'input_y_types': row['input_y_types'],
                'output_X_structure': row['output_X_structure'],
                'output_X_types': row['output_X_types'],
                'output_y_structure': row['output_y_structure'],
                'output_y_types': row['output_y_types'],
                'compatible_scores': [s.strip() for s in row['compatible_scores'].strip('[]').split(',') if s.strip()] if isinstance(row['compatible_scores'], str) else []
            }
        return metadata

    def _get_estimator_params(self, estimator_name):
        # Filtra parâmetros para o estimador específico
        estimator_params_df = self.parameters_df[self.parameters_df['estimator_name'] == estimator_name]
        params = {}
        for _, row in estimator_params_df.iterrows():
            param_name = row['parameter_name']
            param_type = row['parameter_type']
            param_values_str = str(row['parameter_values'])
            
            # Converte string de valores em lista, tratando casos de valores únicos ou None
            if param_values_str == 'None':
                param_values = [None]
            elif param_values_str.startswith('[') and param_values_str.endswith(']'):
                # Remove colchetes e divide por vírgula, tratando espaços
                param_values = [v.strip() for v in param_values_str[1:-1].split(',') if v.strip()]
            else:
                param_values = [param_values_str.strip()]
            
            # Converte os valores para o tipo correto
            processed_values = []
            for val in param_values:
                if val is None:
                    processed_values.append(None)
                elif param_type == 'int':
                    try:
                        processed_values.append(int(val))
                    except ValueError:
                        processed_values.append(val) # Mantém como string se não puder converter
                elif param_type == 'float':
                    try:
                        processed_values.append(float(val))
                    except ValueError:
                        processed_values.append(val) # Mantém como string se não puder converter
                elif param_type == 'bool':
                    processed_values.append(val.lower() == 'true')
                else: # str ou outros
                    processed_values.append(val)
            params[param_name] = processed_values
        return params

    def _build_preprocessor(self, preprocessor_name):
        # Agora usa o estimators_map para instanciar pré-processadores
        preprocessor_class = self.estimators_map.get(preprocessor_name)
        if preprocessor_class:
            # Aqui, no futuro, poderíamos adicionar lógica para passar parâmetros para o pré-processador
            return preprocessor_class()
        return None

    def _check_compatibility(self, prev_output_X_structure, prev_output_X_types, next_input_X_structure, next_input_X_types):
        # Lógica simplificada de compatibilidade. Pode ser expandida.
        # Verifica se a estrutura de saída do anterior é compatível com a entrada do próximo.
        # Por exemplo, se ambos esperam (n_samples, n_features) ou (n_samples,).
        
        # Comparação de estrutura (simplificada)
        # '0' ou 'None' em structure significa que não há restrição ou não se aplica
        structure_compatible = (prev_output_X_structure == '0' or prev_output_X_structure == 'None' or \
                                next_input_X_structure == '0' or next_input_X_structure == 'None' or \
                                prev_output_X_structure == next_input_X_structure)
        
        # Comparação de tipos (simplificada)
        # Se o tipo de saída for um dos tipos de entrada esperados
        types_compatible = False
        if prev_output_X_types == '0' or prev_output_X_types == 'None' or \
            next_input_X_types == '0' or next_input_X_types == 'None': # Wildcard ou não especificado
            types_compatible = True
        else:
            prev_types = [t.strip() for t in prev_output_X_types.strip('[]').split(',') if t.strip()]
            next_types = [t.strip() for t in next_input_X_types.strip('[]').split(',') if t.strip()]
            
            for pt in prev_types:
                if pt in next_types:
                    types_compatible = True
                    break
        
        return structure_compatible and types_compatible

    def create_pipeline(self, steps_config, X_data=None, y_data=None):
        steps = []
        current_output_X_structure = "(n_samples, n_features)" # Assumindo entrada inicial
        current_output_X_types = "float,int" # Assumindo entrada inicial
        
        n_samples = X_data.shape[0] if X_data is not None else None
        n_features = X_data.shape[1] if X_data is not None and X_data.ndim > 1 else None
        n_classes = len(np.unique(y_data)) if y_data is not None else None
        
        for i, (name, estimator_name, params_dict) in enumerate(steps_config):
            estimator_metadata = self.estimator_metadata.get(estimator_name)
            if not estimator_metadata:
                raise ValueError(f"Metadados para o estimador {estimator_name} não encontrados.")

            estimator_class = self.estimators_map.get(estimator_name)
            if not estimator_class:
                raise ValueError(f"Classe do estimador {estimator_name} não encontrada no mapeamento.")

            # 1. Verificar compatibilidade de entrada
            next_input_X_structure = estimator_metadata['input_X_structure']
            next_input_X_types = estimator_metadata['input_X_types']

            if not self._check_compatibility(current_output_X_structure, current_output_X_types, next_input_X_structure, next_input_X_types):
                raise ValueError(
                    f"Incompatibilidade de entrada/saída no passo {i} ({name} - {estimator_name}). "
                    f"Saída anterior: Estrutura '{current_output_X_structure}', Tipos '{current_output_X_types}'. "
                    f"Entrada esperada: Estrutura '{next_input_X_structure}', Tipos '{next_input_X_types}'.")

            # 2. Tratar parâmetros dinâmicos
            processed_params = {}
            for param_name, param_value in params_dict.items():
                if isinstance(param_value, str):
                    if param_value == 'n_samples' and n_samples is not None:
                        processed_params[param_name] = n_samples
                    elif param_value == 'n_features' and n_features is not None:
                        processed_params[param_name] = n_features
                    elif param_value == 'n_classes' and n_classes is not None:
                        processed_params[param_name] = n_classes
                    elif param_value == 'min_samples_per_class' and y_data is not None:
                        processed_params[param_name] = y_data.value_counts().min()
                    elif param_value == 'n_components' and n_features is not None:
                        # Default to min(n_samples, n_features) - 1, or a reasonable fraction
                        val = min(n_samples, n_features) - 1 if n_samples and n_features else None
                        if val is not None and val <= 0: # Ensure n_components is at least 1
                            val = 1
                        processed_params[param_name] = val
                    elif param_value == 'n_clusters' and n_samples is not None:
                        # Default to min(n_samples // 2, n_classes) or a reasonable number
                        val = min(n_samples // 2, n_classes) if n_samples and n_classes else None
                        if val is not None and val <= 0: # Ensure n_clusters is at least 1
                            val = 1
                        processed_params[param_name] = val
                    elif param_value == 'n_output_features' and n_features is not None: # Placeholder for output features
                        # This would typically be determined by the transformer itself, or a subsequent step
                        # For now, we can set it to n_features or a calculated value if context allows
                        processed_params[param_name] = n_features # Example: pass through
                    elif param_value == 'sum_n_components' and n_features is not None: # Placeholder for sum of components
                        # This would be sum of n_components from multiple transformers
                        processed_params[param_name] = n_features # Example: pass through
                    # --- Lógica para variáveis especiais como kernel, sigma, mean ---
                    # O agente precisaria de regras ou um mapeamento para calcular esses valores
                    # ou instanciar classes específicas (e.g., sklearn.gaussian_process.kernels.RBF)
                    # Exemplo (requer importação de RBF se for uma classe):
                    # elif param_name == 'kernel' and param_value == 'RBF':
                    #     from sklearn.gaussian_process.kernels import RBF
                    #     processed_params[param_name] = RBF()
                    # elif param_name == 'sigma' and X_data is not None:
                    #     processed_params[param_name] = np.std(X_data) # Exemplo: desvio padrão dos dados
                    # elif param_name == 'mean' and X_data is not None:
                    #     processed_params[param_name] = np.mean(X_data) # Exemplo: média dos dados
                    elif param_value == 'float':
                        # Get min/max for the specific parameter from parameters_df
                        param_info = self.parameters_df[(self.parameters_df['param_name'] == param_name) & 
                                                        (self.parameters_df['estimators_list'].str.contains(estimator_name, na=False))].iloc[0]
                        min_val = float(param_info['param_min'])
                        max_val = float(param_info['param_max'])
                        processed_params[param_name] = np.random.uniform(min_val, max_val)
                    else:
                        processed_params[param_name] = param_value
                else:
                    processed_params[param_name] = param_value
            
            # Instanciar o estimador com os parâmetros processados
            instance = estimator_class(**processed_params)
            steps.append((name, instance))

            # Atualizar a estrutura e tipos de saída para o próximo passo
            current_output_X_structure = estimator_metadata['output_X_structure']
            current_output_X_types = estimator_metadata['output_X_types']
            
            # Se for o último passo e for um classificador/regressor, a saída y também é relevante
            if i == len(steps_config) - 1:
                self.model = instance # O último estimador é o modelo final
                self.target_column = estimator_metadata['output_y_structure'] # Estrutura do target de saída
                
        self.pipeline = Pipeline(steps)
        return self.pipeline

    def train(self, X, y):
        if self.pipeline is None:
            raise ValueError("Pipeline not created. Call create_pipeline first.")
        self.pipeline.fit(X, y)

    def evaluate(self, X, y):
        if self.pipeline is None:
            raise ValueError("Pipeline not created. Call create_pipeline first.")
        
        y_pred = self.pipeline.predict(X)
        
        metrics = {}
        estimator_name = self.model.__class__.__name__
        compatible_scores = self.estimator_metadata.get(estimator_name, {}).get('compatible_scores', [])

        # Calcula apenas scores compatíveis
        if 'accuracy' in compatible_scores:
            metrics["accuracy"] = accuracy_score(y, y_pred)
        if 'precision' in compatible_scores:
            metrics["precision"] = precision_score(y, y_pred, average='weighted', zero_division=0)
        if 'recall' in compatible_scores:
            metrics["recall"] = recall_score(y, y_pred, average='weighted', zero_division=0)
        if 'f1' in compatible_scores:
            metrics["f1_score"] = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        # Verifica se há mais de uma classe para calcular roc_auc_score e se é compatível
        if len(np.unique(y)) > 1 and 'roc_auc' in compatible_scores:
            try:
                y_proba = self.pipeline.predict_proba(X)
                metrics["roc_auc"] = roc_auc_score(y, y_proba, multi_class='ovr')
            except AttributeError:
                metrics["roc_auc"] = "N/A (Estimator does not support predict_proba)"
        elif 'roc_auc' in compatible_scores:
            metrics["roc_auc"] = "N/A (Single class in target)"

        # Adicionar outros scores de regressão ou clustering conforme necessário e compatível
        estimator_type = self.estimator_metadata.get(estimator_name, {}).get('estimator_type')
        if estimator_type == 'Regressor':
            if 'r2' in compatible_scores:
                metrics["r2_score"] = r2_score(y, y_pred)
            if 'neg_mean_squared_error' in compatible_scores:
                metrics["neg_mean_squared_error"] = mean_squared_error(y, y_pred) # Note: sklearn's neg_mean_squared_error is -MSE
            if 'neg_mean_absolute_error' in compatible_scores:
                metrics["neg_mean_absolute_error"] = mean_absolute_error(y, y_pred) # Note: sklearn's neg_mean_absolute_error is -MAE
        
        return metrics

    def cross_validate(self, X, y, cv=5, scoring='accuracy'):
        if self.pipeline is None:
            raise ValueError("Pipeline not created. Call create_pipeline first.")
        
        n_samples = len(y)
        n_classes = len(np.unique(y))

        if n_classes == 1:
            print("Atenção: Apenas uma classe no target. Usando KFold em vez de StratifiedKFold.")
            cv_strategy = KFold(n_splits=min(cv, n_samples), shuffle=True, random_state=42) # Usar random_state para reprodutibilidade
        else:
            min_samples_per_class = y.value_counts().min()

            # Ajusta n_splits para não ser maior que o número mínimo de amostras por classe
            # ou o número de classes, o que for menor, para evitar o ValueError.
            # Garante que cada fold de teste tenha pelo menos uma amostra de cada classe.
            # O número de splits não pode ser maior que o número de amostras na menor classe.
            # E também não pode ser maior que o número total de amostras.
            adjusted_n_splits = min(cv, min_samples_per_class, n_samples)
            
            if adjusted_n_splits < 2:
                raise ValueError(
                    f"Não é possível realizar validação cruzada com {n_classes} classes e "
                    f"mínimo de {min_samples_per_class} amostras por classe. "
                    f"Considere usar um conjunto de dados com mais amostras por classe ou reduzir o número de splits (cv).")
            
            cv_strategy = StratifiedKFold(n_splits=adjusted_n_splits, shuffle=True, random_state=42)
            print(f"Usando StratifiedKFold com n_splits ajustado para: {adjusted_n_splits}")

        scores = cross_val_score(self.pipeline, X, y, cv=cv_strategy, scoring=scoring)
        return scores
