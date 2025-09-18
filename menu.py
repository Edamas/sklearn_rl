import streamlit as st

# Importa as funções de renderização com nomes únicos de cada página
from academics.gestao_academica import render_gestao_academica
from academics.gestao_tcc import render_gestao_tcc
from academics.pesquisa_academica import render_pesquisa_academica
from academics.formatacao_e_apresentacao import render_formatacao_e_apresentacao

from developers.documentacao_software import render_documentacao_software
from developers.gestao_proposta import render_gestao_proposta
from developers.pesquisa_cientifica import render_pesquisa_cientifica
from developers.desenvolvimento_software import render_desenvolvimento_software

# Importa páginas legadas e a página principal do agente
from A_inputs.A1_datasets import datasets
from C_agent_config.C2_actions.estimators_action import show_estimators
from C_agent_config.C2_actions.parameters_action import show_parameters
from C_agent_config.C2_actions.sklearn_concepts_action import show_sklearn_concepts

def get_pages_config():
    """
    Configura e retorna a estrutura de páginas para a barra lateral do Streamlit.
    """
    pages = {
        "Agente RL": [
            st.Page(datasets, title="Treinamento e Simulações", icon=":material/science:"),
        ],
        "Time Academics": [
            st.Page(render_gestao_academica, title="Gestão Acadêmica", icon=":material/school:"),
            st.Page(render_gestao_tcc, title="Gestão do TCC", icon=":material/assignment:"),
            st.Page(render_pesquisa_academica, title="Pesquisa Acadêmica", icon=":material/science:"),
            st.Page(render_formatacao_e_apresentacao, title="Formatação e Apresentação", icon=":material/draw:"),
        ],
        "Time Developers": [
            st.Page(render_documentacao_software, title="Documentação de Software", icon=":material/description:"),
            st.Page(render_gestao_proposta, title="Gestão da Proposta", icon=":material/assignment:"),
            st.Page(render_pesquisa_cientifica, title="Pesquisa Científica", icon=":material/analytics:"),
            st.Page(render_desenvolvimento_software, title="Desenvolvimento de Software", icon=":material/terminal:"),
        ],
        "A Processar": [
            st.Page(show_estimators, title="Estimadores", icon=":material/table_chart:"),
            st.Page(show_parameters, title="Parâmetros", icon=":material/tune:"),
            st.Page(show_sklearn_concepts, title="Conceitos do Scikit-learn", icon=":material/school:"),
        ],
    }
    return pages

def create_menu():
    """Cria e renderiza o menu de navegação."""
    pages = get_pages_config()
    pg = st.navigation(pages)
    pg.run()
