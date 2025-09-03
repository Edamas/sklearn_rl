import streamlit as st
from docs import show_anotacoes_md, show_cronograma_md, show_cronograma_tsv, show_proposta_tsv, show_rubricas_md, show_tcc_formatado_md, show_readme_md

st.set_page_config(layout="wide")

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
