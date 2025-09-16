import streamlit as st
import pandas as pd

def render_gestao_academica():
    st.title("Gestão Acadêmica")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Esta gestão é responsável pela comunicação com a parte administrativa do grupo, pelo gerenciamento das entregas do TCC no sistema AVA, pela comunicação de resumos e informações importantes para a equipe, e pela organização geral dos arquivos acadêmicos.")
        st.divider()
        st.header("Cronograma e Entregas")
        st.markdown("""| Quinzenas | Início | Atividade | Vencimento | Carência |
| :---: | :---: | :---: | :---: | :---: |
| Quinzena 3 | 08/09/2025 | Primeira entrega | 16/09/2025 | 21/09/2025 |
| Quinzena 6 | 20/10/2025 | Segunda entrega | 28/10/2025 | 03/11/2025 |
| Quinzena 7 | 03/11/2025 | Terceira entrega | 11/11/2025 | 16/11/2025 |""")
        st.divider()
        st.header("Referências e Fontes")
        st.markdown("- Documentação e calendário oficial da disciplina de TCC.")
    with col2:
        st.subheader("Organograma Funcional")
        data = {
            "Time": ["Academics"]*4 + ["Developers"]*4,
            "Função": ["Gestão Acadêmica", "Gestão do TCC", "Pesquisa Acadêmica", "Formatação e Apresentação", "Documentação de Software", "Gestão da Proposta", "Pesquisa Científica", "Desenvolvimento de Software"]
        }
        df = pd.DataFrame(data)
        def highlight_row(row):
            if row.Função == "Gestão Acadêmica": return ['color: white; background-color: #31333F'] * len(row)
            return ['color: black; background-color: white'] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, use_container_width=True)
        st.divider()
        st.subheader("Artefatos")
        st.markdown("##### Inputs\n- Orientações e Rubricas")
        st.markdown("##### Outputs\n- Entregas e Registros no AVA")
        st.divider()
        st.subheader("Requisitos")
        st.markdown("- Gestão de prazos.")
