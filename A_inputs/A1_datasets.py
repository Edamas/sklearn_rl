import streamlit as st
import pandas as pd
from pathlib import Path
import sklearn.datasets as sk_datasets
from functions import df_select_rows, analyze_and_group_columns
from B_input_config.B1_features import feature_definition
from C_agent_config.C1_agent_config import agent_configuration

# Diretório onde os datasets locais estão armazenados.
DATA_DIR = Path("A_inputs/A1_datasets")


def load_dataset(dataset_name: str, load_info: pd.Series):
    """
    Carrega um dataset específico. A função primeiro verifica se é um dataset local
    ou um que precisa ser baixado da biblioteca scikit-learn.

    Args:
        dataset_name (str): O nome do dataset a ser carregado.
        load_info (pd.Series): Metadados sobre o dataset, incluindo tipo e comando de carga.

    Returns:
        pd.DataFrame or None: O DataFrame carregado ou None se ocorrer um erro.
    """
    dataset_type = load_info.get('Tipo', 'sklearn')
    
    # Carrega o dataset de um arquivo local (CSV ou TSV)
    if dataset_type == 'local':
        file_path_str = load_info.get("Comando", dataset_name)
        file_path = DATA_DIR / file_path_str
        
        if not file_path.exists():
            error_message = f"Arquivo do dataset local '{file_path.name}' não encontrado na pasta '{DATA_DIR}'."
            st.error(error_message)
            return None
        
        if file_path.suffix == '.tsv':
            return pd.read_csv(file_path, sep='\t')
        else:
            return pd.read_csv(file_path)
    
    # Tenta carregar uma versão local de um dataset scikit-learn antes de baixar
    local_filename = DATA_DIR / f"{dataset_name}.csv"
    if local_filename.exists():
        return pd.read_csv(local_filename)
    
    # Baixa e salva um dataset do scikit-learn se não existir localmente
    download_command = load_info.get("Comando")
    if pd.isna(download_command) or not download_command:
        error_message = f"Comando de download não encontrado para o dataset '{dataset_name}'."
        st.error(error_message)
        return None
        
    return download_dataset(dataset_name, download_command)

def download_dataset(dataset_name: str, download_command: str):
    """
    Baixa um dataset usando um comando eval do scikit-learn, o converte para
    um DataFrame do pandas e o salva localmente como um arquivo CSV.

    Args:
        dataset_name (str): O nome para salvar o dataset.
        download_command (str): O comando Python para baixar o dataset.

    Returns:
        pd.DataFrame: O dataset baixado como um DataFrame.
    """
    st.toast(f"Dataset '{dataset_name}' não encontrado localmente. Gerando e salvando...")
    # ATENÇÃO: O uso de eval pode ser perigoso se o comando não for confiável.
    # Aqui, é usado para executar comandos de download de datasets do scikit-learn.
    result = eval(download_command, {"datasets": sk_datasets, "pd": pd})

    # Constrói o DataFrame a partir do resultado do download
    if isinstance(result, tuple):
        X, y = result
        df = pd.DataFrame(X, columns=[f"feature{i+1}" for i in range(X.shape[1])])
        if y is not None:
            df["target"] = y
    else:
        df = pd.DataFrame(result, columns=[f"feature{i+1}" for i in range(result.shape[1])])

    # Salva o DataFrame como CSV para uso futuro
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / f"{dataset_name}.csv", index=False, encoding="utf-8")
    st.toast(f"Dataset '{dataset_name}' salvo em {DATA_DIR / f'{dataset_name}.csv'}")
    return df


def update_datasets_metadata():
    """
    Varre o diretório de dados, lê os arquivos CSV e TSV, extrai metadados básicos
    (número de registros, colunas, etc.) e atualiza o arquivo de metadados principal.
    """
    st.toast("Atualizando metadados dos datasets...")
    data_files = list(DATA_DIR.glob('*.csv')) + list(DATA_DIR.glob('*.tsv'))
    
    if not data_files:
        st.warning(f"Nenhum arquivo .csv ou .tsv encontrado na pasta '{DATA_DIR}'.")
        return

    metadata_list = []
    progress_bar = st.progress(0, text="Iniciando atualização de metadados...")
    
    for i, file_path in enumerate(data_files):
        dataset_name = file_path.stem
        progress_text = f"""Processando: {dataset_name} ({i+1}/{len(data_files)})"""
        progress_bar.progress((i + 1) / len(data_files), text=progress_text)
        
        try:
            if file_path.suffix == '.tsv':
                df = pd.read_csv(file_path, sep='\t', low_memory=False)
            else:
                df = pd.read_csv(file_path, low_memory=False)

            n_samples = len(df)
            n_features = len(df.columns)
            
            # Heurística para identificar a coluna alvo
            target_col = next((col for col in df.columns if col.lower() in ["target", "y", "destino"]), None)
            
            metadata_list.append({
                "Dataset": dataset_name,
                "Registros": n_samples,
                "Colunas": n_features,
                "Células Totais": n_samples * n_features,
                "Alvo": target_col,
                "Tipo": "local",
                "Comando": file_path.name
            })
        except Exception as e:
            st.warning(f"Falha ao processar o arquivo {file_path.name}: {e}")

    if metadata_list:
        meta_df = pd.DataFrame(metadata_list)
        meta_df.to_csv(st.session_state.files.get('datasets_metadata'), index=False)
        st.toast("Metadados dos datasets atualizados com sucesso!")
    
    progress_bar.empty()

def datasets():
    """
    Função principal da página "Agente", que corresponde ao passo 1 do fluxo de trabalho.
    Permite ao usuário selecionar um dataset, que então é carregado e processado.
    As seções subsequentes (feature engineering, configuração do agente) são chamadas a partir daqui.
    """
    st.subheader("1. Input")
    st.markdown("### Seleção do Dataset") # This will be the sub-heading for dataset selection

    METADATA_CSV = Path(st.session_state.files.get('datasets_metadata'))

    if st.button("🔄 Atualizar Datasets Locais", help="Busca por novos datasets na pasta de dados e atualiza as estatísticas."):
        update_datasets_metadata()

    try:
        df_meta = pd.read_csv(METADATA_CSV).set_index('Dataset')
    except FileNotFoundError:
        st.warning("Arquivo de metadados não encontrado. Clique em 'Atualizar Datasets' para criá-lo.")
        return
    
    if df_meta.empty:
        st.warning("Nenhum dataset encontrado. Adicione arquivos .csv ou .tsv na pasta de dados e clique em 'Atualizar Datasets'.")
        return

    # Widget para selecionar o dataset
    dataset_name = df_select_rows(df_meta, selection_mode='single-row', key="dataset_selection", prompt="Para começar, selecione um dataset na tabela acima.")
    if not dataset_name:
        return

    

    # Carrega o dataset selecionado
    df = load_dataset(dataset_name, df_meta.loc[dataset_name])
    if df is None:
        return # A função load_dataset já mostra o erro

    st.toast(f"Dataset '{dataset_name}' carregado com sucesso!")

    # Limpa o estado da sessão se o dataset for alterado para evitar inconsistências
    if st.session_state.get("dataset_name") != dataset_name:
        keys_to_clear = ["X_cols", "y_cols", "compatible_estimators", "selected_estimator_names", "num_episodes", "column_summary_df", "task_type"]
        for key in keys_to_clear:
            st.session_state.pop(key, None)
    
    # Armazena informações importantes no estado da sessão
    st.session_state.dataset_name = dataset_name
    st.session_state.original_df = df
    st.session_state.column_summary_df = analyze_and_group_columns(df)
    
    # Chama as próximas seções do fluxo de trabalho em cascata
    # Cada função é responsável por uma etapa e só é chamada se a anterior for concluída.
    if st.session_state.get("dataset_name"):
        st.subheader("2. Processing") # New subheader for processing
        feature_definition() # Passo 2: Feature Engineering
        if st.session_state.get("y_cols") is not None:
            agent_configuration() # Passos 4, 5 e 6: Configuração do Agente
            compatible_estimators = st.session_state.get("compatible_estimators")
            if compatible_estimators is not None and \
				isinstance(compatible_estimators, pd.DataFrame) and \
				not compatible_estimators.empty and \
				st.session_state.get("num_episodes") is not None:
                st.subheader("3. Output") # New subheader for output
                from D_training.D2_training import agent_training
                agent_training() # Passo 7: Treinamento