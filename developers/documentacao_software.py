import streamlit as st
import pandas as pd

def render_documentacao_software():
    st.title("Documentação de Software")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("No time Developers, o membro com foco acadêmico converte a tecnicidade dos desenvolvedores e as especificidades do projeto em termos escritos e encorpados. A revisão final é feita pelo acadêmico do time Academics, mas cada membro revisa a parte do seu 'par' de setor, aplicando suas competências em processos de aplicação diferentes, mas da mesma natureza.")
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
            if row.Função == "Documentação de Software": return ['color: white; background-color: #31333F'] * len(row)
            return ['color: black; background-color: white'] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, width='stretch')
        st.divider()
        st.subheader("Artefatos")
        st.markdown("##### Inputs\n- Docs técnicos e de libs")
        st.markdown("##### Outputs\n- Documentação \"traduzida\"")
