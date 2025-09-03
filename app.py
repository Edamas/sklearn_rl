# app.py
import streamlit as st

# -----------------------------
# Configuração da página (sempre primeiro)
# -----------------------------
APP_TITLE = "Análise de Desempenho de Agente de IA Autônomo (AutoML + RL)"
APP_ICON = "🤖"
PAGE_LAYOUT = "wide"
PAGE_INITIAL_STATE = "expanded"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state=PAGE_INITIAL_STATE
)

# -----------------------------
# Importar funções após set_page_config
# -----------------------------
from docs import (
    show_anotacoes_md, show_cronograma_md, show_cronograma_tsv,
    show_proposta_tsv, show_rubricas_md, show_tcc_formatado_md,
    show_readme_md
)
from sklearn_methods_app import show_sklearn_methods, show_sklearn_categories  # <- adicionado

# -----------------------------
# Título
# -----------------------------
st.header(APP_TITLE, divider='rainbow')

# -----------------------------
# Páginas do app usando st.navigation
# -----------------------------
pages = {
    "Scikit-Learn": [
        st.Page(show_sklearn_categories, title="Categorias de Métodos"),  # <- adicionado
        st.Page(show_sklearn_methods, title="Métodos Scikit-learn")
    ],
    "Documentação": [
        st.Page(show_readme_md, title="README"),
        st.Page(show_anotacoes_md, title="Anotações"),
        st.Page(show_cronograma_md, title="Cronograma (MD)"),
        st.Page(show_cronograma_tsv, title="Cronograma (TSV)"),
        st.Page(show_proposta_tsv, title="Proposta"),
        st.Page(show_rubricas_md, title="Rubricas"),
        st.Page(show_tcc_formatado_md, title="TCC Formatado"),
    ]
}

# -----------------------------
# Navegação
# -----------------------------
pg = st.navigation(pages, position="top")
pg.run()
