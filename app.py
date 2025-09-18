import streamlit as st
import os # Adicionado para manipulação de caminhos

# -----------------------------
# Configuração da página
# -----------------------------
APP_TITLE = "Agente de IA Autônomo para `Sklearn` (AutoML/RL)"
APP_ICON = "🤖"
PAGE_LAYOUT = "wide"
PAGE_INITIAL_STATE = "expanded"


def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout=PAGE_LAYOUT,
        initial_sidebar_state=PAGE_INITIAL_STATE,
    )
    st.header(APP_TITLE, divider='rainbow')

    # Não remover ou alterar esta seção, a menos que saiba exatamente o que está fazendo.
    if 'files' not in st.session_state:
        # Carrega files.tsv no st.session_state.files.
        # ESSENCIAL para o funcionamento dos caminhos de arquivo.
        st.session_state['files'] = {}
        project_root = os.getcwd() # Obtém o diretório raiz do projeto
        with open('files.tsv', 'r', encoding='utf-8') as file:
            files_paths = [name_path.split('\t') for name_path in file.read().split('\n') if name_path.strip()][1:]
            for name, path in files_paths:
                # Converte o caminho relativo em absoluto
                st.session_state['files'][name] = os.path.join(project_root, path)

    # Cria o menu de navegação. ESSENCIAL para a estrutura da aplicação.
    from menu import create_menu  # deve ser importado só após a criação de 'files' na session_state
    create_menu()

if __name__ == '__main__':
    main()  # Não remover ou alterar esta chamada.