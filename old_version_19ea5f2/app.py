import streamlit as st
import sys
import os
import pandas as pd # Added for files.tsv loading
from functions import log_message

# Load files.tsv into st.session_state.files
if 'files' not in st.session_state:
    try:
        files_df = pd.read_csv('D:\\PROGRAMACAO\\sklearn_rl\\files.tsv', sep='\t')
        st.session_state.files = dict(zip(files_df['file_name'], files_df['file_path']))
        # Add 'files' itself to the session state for consistency
        st.session_state.files['files'] = 'D:\\PROGRAMACAO\\sklearn_rl\\files.tsv'
    except FileNotFoundError:
        st.session_state.files = {}
        st.error("Arquivo 'files.tsv' não encontrado. Funcionalidades podem ser limitadas.")
        log_message("ERROR", "Arquivo 'files.tsv' não encontrado.", display_streamlit=False)
    except Exception as e:
        st.session_state.files = {}
        st.error(f"Erro ao carregar 'files.tsv': {e}")
        log_message("EXCEPTION", f"Erro ao carregar 'files.tsv'.", exception=e, display_streamlit=False)

# Get log file path from session state
LOG_FILE_PATH = st.session_state.files.get('log')

# Initialize log_cleared_this_session in st.session_state if not present




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
from D_training.D2_training import agent_training

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
    # log_message("EXCEPTION", "Ocorreu uma exceção não tratada.", exception=exception_obj) # Removido o log para arquivo
    st.error(f"Ocorreu um erro inesperado: {exception_obj}") # Mensagem de erro mais direta
    st.exception(exception_obj) # Exibe o traceback completo na tela

sys.excepthook = handle_exception
