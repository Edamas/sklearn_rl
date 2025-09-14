import streamlit as st
import pandas as pd
from functions import show_log_page, manage_files_df # Assuming show_log_page and manage_files_df are in functions.py

# Import all page functions that will be used in the menu
from A_inputs.A1_datasets import datasets
from B_input_config.B1_features import feature_definition
from C_agent_config.C1_agent_config import agent_configuration
from D_training.D2_training import agent_training
from E_results.E1_results import results
from E_results.E2_graphs import graphs_app
from docs import (
    show_anotacoes_md, show_cronograma_md, show_cronograma_tsv,
    show_proposta_tsv, show_rubricas_md, show_tcc_formatado_md,
    show_readme_md, show_agent_md
)
from sklearn_methods_app import (
    show_sklearn_methods, show_sklearn_categories, show_estimators, show_parameters
)

def show_sitemap_page():
    st.subheader("🗺️ Mapa do Site (Estrutura do Menu)")
    
    sitemap_data = []
    # Iterate through the menu structure to build the sitemap data
    for category, pages_list in get_pages_config().items():
        for page_obj in pages_list:
            
            # Tenta obter o nome da função associada à página
            sitemap_data.append({
                "category": category,
                "title": getattr(page_obj, 'title', None),
                "utl_path": getattr(page_obj, 'url_path', None),
                "Nome da Função": getattr(page_obj, 'func_name', None),
                'default': getattr(page_obj, 'default', False),
            })
    
    df_sitemap = pd.DataFrame(sitemap_data)
    st.dataframe(df_sitemap, width='stretch')
    # Adaptação para Streamlit >= 1.32: o objeto st.Page não expõe diretamente a função.
    # Podemos tentar acessar o atributo 'content' (que é a função associada à página).
    # Se não existir, retorna "N/A".

def get_pages_config():
    pages = {
        "Pipeline em Sklearn": [
            st.Page(datasets, title="Treinamento simples", icon=":material/favorite:"),
            #st.Page(feature_definition, title="2 Definição de Features"),
            #st.Page(agent_configuration, title="3 Configuração do Agente"),
            #st.Page(agent_training, title="4 Treinamento do Agente"),
            #st.Page(results, title="5 Resultados"),
            #st.Page(graphs_app, title="6 Gráficos")
        ],
        "Estudos sobre Scikit-Learn": [
            st.Page(show_estimators, title="Tabela de Estimadores"),
            st.Page(show_parameters, title="Tabela de Parâmetros"),
            st.Page(show_sklearn_methods, title="Métodos Scikit-learn"),
            st.Page(show_sklearn_categories, title="Categorias de Métodos")
        ],
        "TCC - Técnico": [
            st.Page(show_readme_md, title="README"),
            st.Page(show_anotacoes_md, title="Anotações"),
            st.Page(show_agent_md, title="Agente"),
        ],
        "TCC - Acadêmico": [
            st.Page(show_tcc_formatado_md, title="TCC - Relatório 1: Projeto"),
            st.Page(show_rubricas_md, title="TCC: Rubricas de Avaliação Univesp"),
            st.Page(show_proposta_tsv, title="TCC - Proposta"),
            st.Page(show_cronograma_md, title="Cronograma (MD)"),
            st.Page(show_cronograma_tsv, title="Cronograma (TSV)"),
        ],
        "Desenvolvimento": [
            st.Page(show_log_page, title="Visualizar Log"),
            st.Page(manage_files_df, title="Gerenciar Arquivos"), # Added manage_files_df
            st.Page(show_sitemap_page, title="Mapa do Site") 
        ]
    }
    
    return pages
