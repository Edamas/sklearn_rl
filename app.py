import streamlit as st
from docs import show_anotacoes_md, show_cronograma_md, show_cronograma_tsv, show_proposta_tsv, show_rubricas_md, show_tcc_formatado_md, show_readme_md

# -----------------------------
# Configurações principais
# -----------------------------
APP_TITLE = "Análise de Desempenho de Agente de IA Autônomo (AutoML + RL)"
APP_ICON = "🤖"  # pode ser emoji ou caminho para ícone local
PAGE_LAYOUT = "wide"  # opções: "centered", "wide"
PAGE_INITIAL_STATE = "expanded"  # opções: "collapsed", "expanded", "auto"

# -----------------------------
# Configuração da página
# -----------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state=PAGE_INITIAL_STATE
)

# -----------------------------
# Título centralizado
# -----------------------------
st.header(APP_TITLE, divider='rainbow')

# Define the pages dictionary
pages = {
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

# Create the navigation
pg = st.navigation(pages, position="top")

# Run the navigation
pg.run()
