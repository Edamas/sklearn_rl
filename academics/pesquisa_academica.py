import streamlit as st
import pandas as pd

def render_pesquisa_academica():
    st.title("Pesquisa Acadêmica")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Esta gestão é responsável pelo embasamento teórico do trabalho, realizando a busca por livros, referências (primárias e secundárias) e orientações da disciplina, e comunicando-se com a orientadora e a banca.")
        st.divider()
        st.header("Fundamentação Teórica")
        st.markdown("A fundamentação aborda conceitos de agentes inteligentes, sistemas autônomos, AutoML e a biblioteca Scikit-learn, com base em autores como Russell & Norvig (2020), Pedregosa et al. (2011) e Feurer et al. (2019).")
        st.divider()
        st.header("Rúbricas de Pesquisa (Entrega 1 e 2)")
        st.markdown("**Critérios Tecnológicos**\n- PESQUISAR PRINCIPAIS BASES CIENTÍFICAS E BIBLIOTECAS VIRTUAIS\n- PESQUISAR (NAS BASES) POR LIVROS E ARTIGOS\n- PESQUISAR E UTILIZAR FERRAMENTAS DE GERENCIAMENTO DE REFERÊNCIAS\n\n**Critérios Investigativos**\n- Delinear a área de estudo\n- Apresentar os principais autores sobre o tema\n- Relacionar as principais referências com o tema")
        st.divider()
        st.header("Referências e Fontes")
        st.markdown("""- ABNT – NBR 14724.
- FEURER, M. et al. Auto-sklearn 2.0.
- PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python.
- RUSSELL, S.; NORVIG, P. Artificial Intelligence: A Modern Approach.
- ZOPH, B.; LE, Q. V. Neural Architecture Search with Reinforcement Learning.""")
    with col2:
        st.subheader("Organograma Funcional")
        data = {
            "Time": ["Academics"]*4 + ["Developers"]*4,
            "Função": ["Gestão Acadêmica", "Gestão do TCC", "Pesquisa Acadêmica", "Formatação e Apresentação", "Documentação de Software", "Gestão da Proposta", "Pesquisa Científica", "Desenvolvimento de Software"]
        }
        df = pd.DataFrame(data)
        def highlight_row(row):
            if row.Função == "Pesquisa Acadêmica": return ['color: white; background-color: #31333F'] * len(row)
            return ['color: black; background-color: white'] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, use_container_width=True)
        st.divider()
        st.subheader("Artefatos")
        st.markdown("##### Inputs\n- Bibliografia do curso")
        st.markdown("##### Outputs\n- Embasamento teórico")
        st.divider()
        st.subheader("Requisitos")
        st.markdown("- Pesquisar bases científicas.\n- Usar gerenciador de referências.")
