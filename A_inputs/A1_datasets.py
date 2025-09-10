import streamlit as st
import pandas as pd
from pathlib import Path
import sklearn.datasets as sk_datasets
from functions import df_select_rows, log_message, build_feature_table
from B_input_config.B1_features import feature_definition
from C_agent_config.C1_agent_config import agent_configuration
from D_training.D1_training import agent_training

DATA_DIR = Path("A_inputs/A1_datasets")
ESTIMATORS_FILE = st.session_state.files.get('estimators')

def load_dataset(dataset_name: str, load_info: pd.Series):
    """Carrega um dataset local ou baixa um do scikit-learn."""
    dataset_type = load_info.get('Tipo', 'sklearn')
    
    if dataset_type == 'local':
        file_path_str = load_info.get("Comando", dataset_name)
        file_path = DATA_DIR / file_path_str
        
        if not file_path.exists():
            log_message("ERROR", f"Arquivo do dataset local '{file_path.name}' não encontrado na pasta '{DATA_DIR}'.")
            return None
        
        try:
            if file_path.suffix == '.tsv':
                return pd.read_csv(file_path, sep='\t')
            else:
                return pd.read_csv(file_path)
        except Exception as e:
            log_message("EXCEPTION", f"Erro ao ler o arquivo '{file_path.name}'.", exception=e)
            return None
    
    # Lógica para datasets do scikit-learn
    local_filename = DATA_DIR / f"{dataset_name}.csv"
    if local_filename.exists():
        return pd.read_csv(local_filename)
    
    download_command = load_info.get("Comando")
    if pd.isna(download_command) or not download_command:
        log_message("ERROR", f"Comando de download não encontrado para o dataset '{dataset_name}'.")
        return None
        
    return download_dataset(dataset_name, download_command)

def download_dataset(dataset_name: str, download_command: str):
    st.toast(f"Dataset '{dataset_name}' não encontrado. Gerando...")
    try:
        result = eval(download_command, {"datasets": sk_datasets, "pd": pd})
    except Exception as e:
        log_message("EXCEPTION", f"Erro ao executar comando para download do dataset '{dataset_name}'.", exception=e)
        return None

    if isinstance(result, tuple):
        X, y = result
        df = pd.DataFrame(X, columns=[f"feature{i+1}" for i in range(X.shape[1])])
        if y is not None:
            df["target"] = y
    else:
        df = pd.DataFrame(result, columns=[f"feature{i+1}" for i in range(result.shape[1])])

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / f"{dataset_name}.csv", index=False, encoding="utf-8")
    st.toast(f"Dataset '{dataset_name}' gerado e salvo em {DATA_DIR / f'{dataset_name}.csv'}")
    return df


def update_datasets_metadata():
    """
    Varre a pasta 'data', lê os arquivos CSV e TSV, extrai metadados
    e salva no arquivo 'datasets_metadata.csv' na raiz do projeto.
    """
    st.toast("Atualizando metadados dos datasets...")
    data_files = list(DATA_DIR.glob('*.csv')) + list(DATA_DIR.glob('*.tsv'))
    
    if not data_files:
        log_message("WARNING", f"Nenhum arquivo .csv ou .tsv encontrado na pasta '{DATA_DIR}'.")
        return

    metadata_list = []
    progress_bar = st.progress(0, text="Iniciando atualização...")
    
    for i, file_path in enumerate(data_files):
        dataset_name = file_path.stem
        progress_text = "Processando: " + str(dataset_name) + " (" + str(i+1) + " de " + str(len(data_files)) + ")"
        progress_bar.progress((i + 1) / len(data_files), text=progress_text)
        
        try:
            if file_path.suffix == '.tsv':
                df = pd.read_csv(file_path, sep='\t', low_memory=False)
            else:
                df = pd.read_csv(file_path, low_memory=False)

            n_samples = len(df)
            n_features = len(df.columns)
            total_cells = n_samples * n_features
            
            # Lógica simples para identificar o alvo
            target_col = None
            for col in ["target", "y", "destino"]:
                if col in [c.lower() for c in df.columns]:
                    target_col = col
                    break
            
            metadata_list.append({
                "Dataset": dataset_name,
                "Registros": n_samples,
                "Colunas": n_features,
                "Células Totais": total_cells,
                "Alvo": target_col,
                "Tipo": "local",
                "Comando": file_path.name
            })

        except Exception as e:
            log_message("EXCEPTION", f"Erro ao processar o arquivo '{file_path.name}'.", exception=e)

    if metadata_list:
        meta_df = pd.DataFrame(metadata_list)
        meta_df.to_csv(st.session_state.files.get('datasets_metadata'), index=False)
        st.toast("Metadados dos datasets atualizados com sucesso!")
    
    progress_bar.empty()

def datasets():
    st.subheader("📚 1. Seleção do Dataset")

    METADATA_CSV = Path(st.session_state.files.get('datasets_metadata'))

    if st.button("🔄 Atualizar Datasets", help="Busca por novos datasets na pasta 'data' e atualiza as estatísticas."):
        update_datasets_metadata()

    if not METADATA_CSV.exists():
        st.info("Arquivo de metadados não encontrado. Clique em 'Atualizar Datasets' para gerá-lo.")
        st.stop()

    df_meta = pd.read_csv(METADATA_CSV).set_index('Dataset')
    
    if df_meta.empty:
        log_message("WARNING", "Nenhum dataset encontrado. Adicione arquivos .csv ou .tsv na pasta 'data' e clique em 'Atualizar Datasets'.")
        st.stop()

    dataset_name = df_select_rows(df_meta, selection_mode='single-row', key="dataset_selection") # Added key
    if not dataset_name:
        st.info("Para começar, selecione um dataset na tabela.")
        st.stop()

    df = load_dataset(dataset_name, df_meta.loc[dataset_name])
    
    if df is None:
        log_message("ERROR", f"Não foi possível carregar o dataset '{dataset_name}'.")
        st.stop()

    st.success(f"Dataset '{dataset_name}' carregado com sucesso!")

    # If dataset has changed, clear subsequent selections
    if st.session_state.get("dataset_name") != dataset_name:
        keys_to_clear = ["X_cols", "y_cols", "compatible_estimators", "selected_estimator_names", "num_episodes"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
    
    st.session_state.dataset_name = dataset_name
    st.session_state.original_df = df
    
    st.write(f"DEBUG: Dataset selected: {st.session_state.dataset_name}")