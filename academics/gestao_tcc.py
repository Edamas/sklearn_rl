import streamlit as st
import pandas as pd

def render_gestao_tcc():
    st.title("Gestão do TCC")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Esta gestão define as estratégias e o escopo do projeto, incluindo a justificativa, os objetivos, as inovações, o plano e o cronograma, atuando com um modelo de liderança servidora e democrática.")
        st.divider()
        st.header("Proposta do Projeto")
        st.markdown("""**Resumo:** O objetivo é analisar o desempenho de agentes de IA autônomos na utilização da suíte Scikit-learn em projetos de AutoML, visando automatizar etapas como preparação de dados e seleção de algoritmos para otimizar tempo e recursos.

**Justificativa:** O campo do Aprendizado de Máquina (ML) é complexo. A automação (AutoML) surge para democratizar seu acesso. Este trabalho propõe o uso de Aprendizado por Reforço (RL) para preencher a lacuna de abordagens mais inteligentes em AutoML.""")
        st.divider()
        st.header("Referências e Fontes")
        st.markdown("- N/A")
    with col2:
        st.subheader("Organograma Funcional")
        data = {
            "Time": ["Academics"]*4 + ["Developers"]*4,
            "Função": ["Gestão Acadêmica", "Gestão do TCC", "Pesquisa Acadêmica", "Formatação e Apresentação", "Documentação de Software", "Gestão da Proposta", "Pesquisa Científica", "Desenvolvimento de Software"]
        }
        df = pd.DataFrame(data)
        def highlight_row(row):
            if row.Função == "Gestão do TCC": return ['color: white; background-color: #31333F'] * len(row)
            return ['color: black; background-color: white'] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, use_container_width=True)
        st.divider()
        st.subheader("Artefatos")
        st.markdown("##### Inputs\n- Estratégias de Relatório")
        st.markdown("##### Outputs\n- Objetivos e Justificativas")
        st.divider()
        st.subheader("Requisitos")
        st.markdown("- Delinear área de estudo.\n- Apresentar autores.")
