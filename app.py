import streamlit as st
import sys
import os
import pandas as pd # Added for files.tsv loading
from functions import log_message

# Load files.tsv into st.session_state.files
if 'files' not in st.session_state:
    try:
        # Construct path relative to this script file for robustness
        script_dir = os.path.dirname(os.path.abspath(__file__))
        files_tsv_path = os.path.join(script_dir, 'files.tsv')
        
        files_df = pd.read_csv(files_tsv_path, sep='\t')
        
        # Create absolute paths for all files listed in files.tsv
        st.session_state.files = {}
        for index, row in files_df.iterrows():
            file_name = row['file_name']
            relative_path = row['file_path']
            # Ensure forward slashes are used for cross-platform compatibility
            absolute_path = os.path.join(script_dir, *relative_path.split('/'))
            st.session_state.files[file_name] = absolute_path

        # Add 'files' itself to the session state for consistency
        st.session_state.files['files'] = files_tsv_path
        
    except FileNotFoundError:
        st.session_state.files = {}
        st.error("Arquivo 'files.tsv' não encontrado. Verifique se o arquivo existe no diretório raiz do projeto.")
        log_message("ERROR", "Arquivo 'files.tsv' não encontrado.", display_streamlit=False)
        st.stop()
    except Exception as e:
        st.session_state.files = {}
        st.error(f"Erro ao carregar 'files.tsv': {e}")
        log_message("EXCEPTION", f"Erro ao carregar 'files.tsv'.", exception=e, display_streamlit=False)
        st.stop()

# Get log file path from session state
LOG_FILE_PATH = st.session_state.files.get('log')

# Initialize log_cleared_this_session in st.session_state if not present
if 'log_cleared_this_session' not in st.session_state:
    st.session_state.log_cleared_this_session = False

# Clear log.tsv only once per session
if not st.session_state.log_cleared_this_session:
    if LOG_FILE_PATH and os.path.exists(LOG_FILE_PATH):
        try:
            with open(LOG_FILE_PATH, "w") as f:
                f.truncate(0)
            log_message("INFO", "Log file cleared on app start/refresh.", display_streamlit=False)
            st.session_state.log_cleared_this_session = True # Mark as cleared
        except Exception as e:
            log_message("EXCEPTION", f"Erro ao limpar o arquivo de log '{LOG_FILE_PATH}'.", exception=e)
            st.session_state.log_cleared_this_session = True # Mark as cleared even on error to prevent repeated attempts



# Clear any lingering page navigation state
if 'current_page_key' in st.session_state:
    del st.session_state['current_page_key']
if 'StreamlitAPIException' in st.session_state: # Clear any previous error flags
    del st.session_state['StreamlitAPIException']

# -----------------------------
# Configuração da página
# -----------------------------
APP_TITLE = "Análise de Desempenho de Agente de IA Autônomo (AutoML + RL)"
APP_ICON = "🤖"
PAGE_LAYOUT = "wide"
PAGE_INITIAL_STATE = "expanded"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state=PAGE_INITIAL_STATE,
    
)

# Importações das funções das páginas
from A_inputs.A1_datasets import datasets
from B_input_config.B1_features import feature_definition
from C_agent_config.C1_agent_config import agent_configuration
from D_training.D1_training import agent_training

# -----------------------------
# Título
# -----------------------------
st.header(APP_TITLE, divider='rainbow')

# -----------------------------
# Execução sequencial das seções da página
# -----------------------------
datasets()
feature_definition()
agent_configuration()
agent_training()

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't log KeyboardInterrupt
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    exception_obj = exc_value if isinstance(exc_value, Exception) else Exception(str(exc_value))
    log_message("EXCEPTION", "Ocorreu uma exceção não tratada.", exception=exception_obj)
    st.error(f"Ocorreu um erro inesperado. Por favor, verifique o arquivo {st.session_state.files.get('log')} para mais detalhes.")

sys.excepthook = handle_exception
