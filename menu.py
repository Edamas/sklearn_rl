import streamlit as st
import pandas as pd
from A_inputs.A1_datasets import datasets
from C_agent_config.C2_actions.estimators_action import show_estimators
from C_agent_config.C2_actions.parameters_action import show_parameters
from C_agent_config.C2_actions.sklearn_concepts_action import show_sklearn_concepts

import os

##################################
# Menu e Páginas
##################################

def show_docs(file_path):
    """
    Displays the content of a file in the Streamlit app, using the appropriate widget based on the file extension.
    """
    if file_path.endswith(".md"):
        # Special handling for CRONOGRAMA.md if it needs to be a dataframe
        if os.path.basename(file_path) == "CRONOGRAMA.md":
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Find the table lines
            table_lines = []
            in_table = False
            for line in lines:
                if line.strip().startswith('|') and line.strip().endswith('|'):
                    in_table = True
                    table_lines.append(line.strip())
                elif in_table:
                    break # End of table
            
            if len(table_lines) >= 2:
                # Parse header
                header = [h.strip() for h in table_lines[0].strip('|').split('|')]
                
                # Parse data
                data = []
                for line in table_lines[2:]: # Skip header and separator
                    row = [d.strip() for d in line.strip('|').split('|')]
                    data.append(row)
                
                df = pd.DataFrame(data, columns=header)
                st.dataframe(df, width="stretch")   # <--- substitui use_container_width=True
            else:
                # Fallback to markdown if table not found or malformed
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                st.markdown(content, unsafe_allow_html=True)

        else: # Regular markdown file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            st.markdown(content, unsafe_allow_html=True)

    elif file_path.endswith(".tsv"):
        df = pd.read_csv(file_path, sep='\t')
        st.dataframe(df, width="stretch")   # <--- substitui use_container_width=True
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.text_area("Content", content, height=400)
    elif file_path.endswith(".jpg"):
        st.image(file_path)
    else:
        st.warning(f"File type not supported for: {file_path}")


# Functions for each specific document
def show_anotacoes_md():        show_docs(st.session_state['files'].get('ANOTACOES', 'docs/ANOTACOES.md'))
def show_cronograma_tsv():      show_docs(st.session_state['files'].get('cronograma', 'docs/cronograma.tsv'))
def show_proposta_tsv():        show_docs(st.session_state['files'].get('proposta', 'docs/proposta.tsv'))
def show_rubricas_md():         show_docs(st.session_state['files'].get('rubricas', 'docs/rubricas.md'))
def show_tcc_formatado_md():    show_docs(st.session_state['files'].get("TCC_FORMATADO", "docs/TCC_FORMATADO.md"))
def show_agent_md():            show_docs(st.session_state['files'].get('agent.md', 'docs/agent.md'))
def show_readme_md():           show_docs(st.session_state['files'].get('readme', 'readme.md'))

def get_pages_config():
    pages = {
        "1. Agente": [
            st.Page(show_agent_md, title="agent.md", icon=":material/notes:"),
        ],
        "2. Simulações": [
            st.Page(datasets, title="2.1 Agente (random search)", icon=":material/smart_toy:"),
        ],
        "3. Ações": [
            st.Page(show_estimators, title="Tabela de Estimadores", icon=":material/table_chart:"),
            st.Page(show_parameters, title="Tabela de Parâmetros", icon=":material/tune:"),
            st.Page(show_sklearn_concepts, title="Conceitos Scikit-learn", icon=":material/school:"),
        ],
        "4. Ambiente": [
            
        ],
        "5. TCC - Acadêmico": [
            st.Page(show_readme_md, title="README", icon=":material/description:"),
            st.Page(show_anotacoes_md, title="Discussão Atual", icon=":material/forum:"),
            st.Page(show_tcc_formatado_md, title="Entrega 1 - Projeto", icon=":material/assignment:"),
            st.Page(show_rubricas_md, title="Critérios de Avaliação", icon=":material/grading:"),
            st.Page(show_cronograma_tsv, title="Cronograma Univesp", icon=":material/calendar_month:"),
            st.Page(show_proposta_tsv, title="Proposta Inicial", icon=":material/lightbulb:"),
        ],
    }
    
    return pages

def create_menu():
    # não alterar esta função
    pages = get_pages_config()
    pg = st.navigation(pages, position='sidebar', expanded=True)
    pg.run()

def main():
    pass

if __name__ == '__main__':
    main()